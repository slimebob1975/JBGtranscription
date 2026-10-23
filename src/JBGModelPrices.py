"""
Modellmenyn och priserna för OpenAI-modellerna, från policy/model_prices.json.

Filen läses en gång per process, alltså vid varje omstart, och kontrolleras
då av validate_price_table(). OpenAI har inget API för priser, så de förs in
för hand från prislistan; kontrollen varnar när ett pris saknas eller när
datumet för senaste avstämning är gammalt.

Kostnaden räknas per anrop:
    (prompt - cachade) * in + cachade * cachad + svar * ut, per miljon tokens.
Resonemangstokens ingår i svarstokens och debiteras som utdata.
"""

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

PRICES_PATH = Path(__file__).resolve().parents[1] / "policy" / "model_prices.json"

# Varning när priserna inte har stämts av på så här många dagar.
MAX_PRICE_AGE_DAYS = 90

# Används bara om filen saknas eller inte går att läsa, så att appen ändå
# går att använda. Kostnad visas då inte.
FALLBACK_MENU = ["gpt-5.2"]

_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_PRICE_FIELDS = ("in", "cachad", "ut")


def read_price_table(path: Path = PRICES_PATH) -> dict:
    """Read the file into a dict; problems are listed under "errors" rather than raised."""
    table = {"currency": "USD", "checked": None, "menu": [], "default": "",
             "prices": {}, "errors": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        table["errors"].append(f"Prisfilen {path.name} kunde inte läsas: {e}")
        table["menu"] = list(FALLBACK_MENU)
        table["default"] = FALLBACK_MENU[0]
        return table

    table["currency"] = data.get("_valuta") or "USD"
    table["checked"] = data.get("_kontrollerad") or None
    menu = data.get("_i_menyn")
    table["menu"] = [m for m in menu if isinstance(m, str) and m] if isinstance(menu, list) else []
    table["default"] = data.get("_standardmodell") or ""

    for name, entry in data.items():
        if name.startswith("_"):
            continue
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(f), (int, float)) and entry.get(f) >= 0 for f in _PRICE_FIELDS
        ):
            table["errors"].append(
                f"Priset för {name} är ofullständigt; det behöver in, cachad och ut som tal."
            )
            continue
        table["prices"][name] = {f: float(entry[f]) for f in _PRICE_FIELDS}

    if not table["menu"]:
        table["errors"].append("_i_menyn saknas eller är tom; menyn visar bara reservmodellen.")
        table["menu"] = list(FALLBACK_MENU)
    if table["default"] not in table["menu"]:
        table["default"] = table["menu"][0]
    return table


@lru_cache(maxsize=1)
def get_price_table() -> dict:
    return read_price_table()


def price_for(model: str, table: dict = None):
    """The price entry for a model name, or None.

    An exact name wins; otherwise a dated name such as "gpt-5.2-2025-12-11"
    matches "gpt-5.2". No looser matching: "gpt-5.4-mini" must never fall
    back to the price of "gpt-5.4".
    """
    prices = (table or get_price_table())["prices"]
    if model in prices:
        return prices[model]
    return prices.get(_DATE_SUFFIX.sub("", model or ""))


def call_cost(model: str, sent: int, cached: int, received: int, table: dict = None):
    """Cost of one call in the table's currency, or None if the model has no price."""
    price = price_for(model, table)
    if price is None:
        return None
    sent, cached, received = sent or 0, cached or 0, received or 0
    cached = min(cached, sent)
    return (
        (sent - cached) * price["in"]
        + cached * price["cachad"]
        + received * price["ut"]
    ) / 1_000_000


def validate_price_table(table: dict = None, today: date = None) -> list:
    """Warnings about the table, for the log at startup."""
    table = table or get_price_table()
    today = today or date.today()
    warnings = list(table["errors"])

    for model in table["menu"]:
        if price_for(model, table) is None:
            warnings.append(f"Modellen {model} finns i menyn men saknar pris; kostnad visas inte för den.")

    checked = table["checked"]
    if not checked:
        warnings.append(
            "_kontrollerad saknas i prisfilen. Stäm av priserna mot OpenAI:s prislista och ange datumet."
        )
    else:
        try:
            age = (today - date.fromisoformat(checked)).days
            if age > MAX_PRICE_AGE_DAYS:
                warnings.append(
                    f"Priserna stämdes senast av {checked}, för {age} dagar sedan. "
                    "Kontrollera dem mot OpenAI:s prislista."
                )
        except ValueError:
            warnings.append(f"_kontrollerad={checked!r} är inte ett datum på formen ÅÅÅÅ-MM-DD.")
    return warnings


def format_cost(value: float) -> str:
    """Swedish decimal comma; small amounts get more decimals so they do not show as 0,00."""
    text = f"{value:.2f}" if value >= 0.01 or value == 0 else f"{value:.4f}"
    return text.replace(".", ",")
