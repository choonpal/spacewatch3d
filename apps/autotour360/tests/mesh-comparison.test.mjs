import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as context from '../mesh-change-context.js';

test('a failed 3D renderer cannot expose completed comparison statistics through navigation', async () => {
  const source = (await readFile(new URL('../mesh-comparison.js', import.meta.url), 'utf8')).replace(/^import .*;\r?\n/, '');
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      id, dataset: {}, hidden: id === 'retry', disabled: false, textContent: '',
      listeners: {}, classList: { add() {} }, children: [],
      addEventListener(type, listener) { this.listeners[type] = listener; },
      querySelectorAll() { return []; }, replaceChildren() { this.children = []; },
      // A renderer compilation/context failure occurs before map initialization.
      getContext() { return null; },
    });
    return elements.get(id);
  }
  const controls = ['before', 'after', 'overview', 'top', 'mode', 'cut'].map(element);
  const result = { kind: 'geometry_method_comparison', scene: 'Rescan test · 0 → 2', core_count: 12,input_hashes:['a','b'],
    displays: [{ url: '/first.bin',count:0 }, { url: '/second.bin',count:0 }], regions: [] };
  const reference = {kind:'rescan_object_changes',scene:'test',source_epochs:[0,2],keyframes:[],
    regions:[{id:1,type:'added',bounds:{min:[0,0,0],max:[1,1,1]},keyframes:[[],[]]}]};
  const independent={kind:'m3c2_independent_changes',scene:'test',source_epochs:[0,2],source_hashes:['a','b'],label_independent:true,
    classification_uses_instance_ids:false,regions:[],keyframes:[],region_masks:[{url:'/mask0',count:0},{url:'/mask1',count:0}]};
  const document = {
    getElementById: element,
    querySelectorAll(selector) { return selector.startsWith('.toolbar') ? controls : []; },
  };
  const errors = [];
  vm.runInNewContext(source, {
    ...context, document, window: {}, console: { error: (...args) => errors.push(args) },
    fetch: async url => ({ ok: true, json: async () => url==='/rescan/changes.json'?reference:url==='/m3c2-independent/changes.json'?independent:result, arrayBuffer: async () => new ArrayBuffer(0) }),
  });
  for (let n = 0; n < 20 && element('status').dataset.state === 'loading'; n++) await new Promise(setImmediate);
  assert.equal(element('status').dataset.state, 'error');
  assert.equal(element('retry').hidden, false);
  assert.equal(errors.length, 1);
  assert.match(errors[0][1].message,/WebGL/);
  assert.ok(controls.every(control => control.disabled));
  // This was the misleading path: a click filled percentages despite blank maps.
  element('overview').listeners.click();
  element('before').listeners.click();
  assert.equal(element('stats-0').children.length, 0);
  assert.equal(element('stats-1').children.length, 0);
  assert.equal(element('status').dataset.state, 'error');
});
