// Run with: node --test tests/test_processing_overlay.js
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/javascript/script.js'), 'utf8');
const lifecycle = source.slice(source.indexOf('// The timer covers preparation'));

function setup(responses, { encrypted = false, missingFile = false, cryptoFailure = false } = {}) {
    const nodes = new Map();
    function node(id) {
        if (!nodes.has(id)) nodes.set(id, { textContent: '', value: '', checked: false,
            disabled: false, hidden: false, attrs: {}, style: {},
            setAttribute(k, v) { this.attrs[k] = v; },
            removeAttribute(k) { delete this.attrs[k]; }, remove() {}, click() {} });
        return nodes.get(id);
    }
    const file = new Blob(['audio']);
    file.name = 'test.wav';
    node('audioFile').files = missingFile ? [] : [file];
    node('enableEncryption').checked = encrypted;
    node('accuracySelect').selectedOptions = [{ textContent: 'medel' }];
    node('accuracySelect').value = 'medel';
    node('spinner-container').hidden = true;
    const panel = node('panel');
    const calls = [], timers = new Map(), alerts = [];
    let now = 1000, nextTimer = 0, downloads = 0;
    const crypto = cryptoFailure ? { subtle: { generateKey: async () => { throw Error('crypto'); } } } : require('node:crypto').webcrypto;
    const context = vm.createContext({
        document: { getElementById: node, querySelectorAll: s => s.includes('fieldset') ? [panel] : [node('radio')],
            querySelector: () => ({ value: 'raw' }), body: { appendChild() {} },
            createElement: () => ({ style: {}, click() { downloads++; }, remove() {} }) },
        performance: { now: () => now }, setInterval: fn => { timers.set(++nextTimer, fn); return nextTimer; },
        clearInterval: id => timers.delete(id), setTimeout: fn => { queueMicrotask(fn); },
        FormData, Blob, Uint8Array, btoa, crypto, window: { crypto },
        URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} },
        console: { log() {} }, localStorage: { setItem() {} }, alert: m => alerts.push(m),
        supportedAudioExtensions: ['.wav'], maxUploadMb: 1024, globalEncryptionKeyBase64: '',
        currentSummaryStyle: () => '', summaryPromptText: '', questionsPromptText: '', speakersPromptText: '', customPromptText: '',
        fetch: async (url, opts) => {
            calls.push({ url, opts, detail: node('processingDetail').textContent, hidden: node('spinner-container').hidden });
            assert.equal(node('spinner-container').hidden, false);
            assert.equal(timers.size, 1);
            const response = responses.shift();
            assert.ok(response, 'unexpected fetch');
            if (response instanceof Error) throw response;
            if (response.wait) await response.wait;
            return { ok: response.status < 400, status: response.status,
                json: async () => { if (response.badJson) throw Error('JSON'); return response.data; },
                blob: async () => new Blob(['docx']), text: async () => '' };
        },
    });
    vm.runInContext(lifecycle, context);
    return { node, panel, calls, timers, alerts, context, upload: () => vm.runInContext('uploadFile()', context),
        tick: ms => { now += ms; for (const fn of timers.values()) fn(); }, downloads: () => downloads };
}
const uploaded = () => ({ status: 200, data: { file_id: 'job' } });
const pending = () => ({ status: 202, data: { done: false, status: 'Analyserar texten' } });
const done = () => ({ status: 200, data: { done: true, download_filename: 'resultat.docx' } });
function clean(s) {
    assert.equal(s.node('spinner-container').hidden, true);
    assert.equal(s.timers.size, 0);
    assert.equal(s.panel.inert, false);
    assert.equal(s.panel.attrs['aria-busy'], undefined);
}
test('overlay, elapsed time, live status, duplicate submission, encrypted download', async () => {
    let release;
    const wait = new Promise(r => { release = r; });
    const s = setup([uploaded(), pending(), done(), { status: 200, wait }], { encrypted: true });
    const work = s.upload();
    await s.upload();
    assert.equal(s.node('processingAccuracy').textContent, 'Vald noggrannhet: medel');
    assert.equal(s.panel.attrs['aria-busy'], 'true');
    s.tick(84000);
    assert.equal(s.node('processingElapsed').textContent, '01:24');
    // WebCrypto runs asynchronously, so wait for the mocked download request.
    for (let i = 0; s.calls.length < 4 && i < 1000; i++) await new Promise(r => setImmediate(r));
    assert.deepEqual(s.calls.map(c => c.url), ['/upload/', '/transcription/job', '/transcription/job', '/download/job']);
    assert.equal(s.calls[2].detail, 'Analyserar texten');
    assert.equal(s.calls[3].detail, 'Hämtar resultatfilen…');
    const key = s.calls[0].opts.body.get('encryption_key');
    assert.ok(key);
    assert.equal(s.calls[3].opts.method, 'POST');
    assert.equal(s.calls[3].opts.body.get('encryption_key'), key);
    assert.equal(s.node('spinner-container').hidden, false);
    release(); await work;
    assert.equal(s.downloads(), 1); clean(s);
});
test('validation does not start overlay', async () => {
    const s = setup([], { missingFile: true }); await s.upload();
    assert.equal(s.alerts.length, 1); assert.equal(s.calls.length, 0);
    assert.equal(s.node('spinner-container').hidden, true); assert.equal(s.timers.size, 0);
});
for (const [name, responses, opts, expected] of [
    ['crypto failure', [], { encrypted: true, cryptoFailure: true }, /Kunde inte/],
    ['upload HTTP error', [{ status: 413 }], {}, /413/],
    ['invalid upload JSON', [{ status: 200, badJson: true }], {}, /Kunde inte/],
    ['missing job id', [{ status: 200, data: {} }], {}, /Kunde inte/],
    ['status HTTP error', [uploaded(), { status: 500 }], {}, /500/],
    ['three failed polls', [uploaded(), Error('network'), Error('network'), Error('network')], {}, /fortfarande pågå/],
    ['download HTTP error', [uploaded(), done(), { status: 502 }], {}, /502/],
]) test(name, async () => {
    const s = setup(responses, opts); await s.upload(); clean(s);
    assert.match(s.node('status').textContent, expected); assert.equal(s.downloads(), 0);
});
test('successful poll resets failure counter', async () => {
    const s = setup([uploaded(), Error('network'), Error('network'), pending(), Error('network'), done(), { status: 200 }]);
    await s.upload(); clean(s); assert.equal(s.downloads(), 1);
});
