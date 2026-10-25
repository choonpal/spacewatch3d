import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const root = process.cwd();
const directory = fs.existsSync(path.join(root, 'outputs/scenediff-mesh-rgb-20261024/focus-comparison-20261025'))
  ? 'outputs/scenediff-mesh-rgb-20261024/focus-comparison-20261025'
  : 'src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/focus-comparison-20261025';
const data = JSON.parse(fs.readFileSync(path.join(root, directory, 'panorama-review.json'), 'utf8'));
const interaction = JSON.parse(fs.readFileSync(path.join(root, directory, 'panorama-interactions.json'), 'utf8'));
const html = fs.readFileSync(path.join(root, 'tools/scenediff_mesh/panorama_review.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'tools/scenediff_mesh/panorama_review.js'), 'utf8')
  .replace(/^import .*$/gm, '')
  .replace(/\ninit\(\)\.catch\([\s\S]*$/, '\n');

function defaultValue(id) {
  const select = html.match(new RegExp(`<select[^>]*id="${id}"[^>]*>([\\s\\S]*?)<\\/select>`))[1];
  const options = [...select.matchAll(/<option\b([^>]*)>(.*?)<\/option>/g)];
  return (options.find(x => /\bselected\b/.test(x[1])) || options[0])[1].match(/value="([^"]*)"/)[1];
}
function selected(filter = defaultValue('filter'), exclude = true, zone = 'R01') {
  const fields = {'#zone': {value: zone}, '#filter': {value: filter}, '#exclude-person': {checked: exclude}};
  const context = vm.createContext({document: {querySelector: selector => fields[selector]}, __data: data, __interaction: interaction});
  vm.runInContext(script + '\ndata=__data;interaction=__interaction;globalThis.answer=data.cases.filter(matches).map(c=>c.id);', context);
  return Array.from(context.answer);
}
const emptyPair = 'R01--D2-f133--D3-f142';
const remainingPair = 'R01--D2-f130--D3-f136';

test('default list omits R01 133/142 whose 156 pixels are all excluded', () => {
  assert(!selected().includes(emptyPair));
  assert(selected().includes(remainingPair));
});
test('candidate selection follows the actual person exclusion toggle', () => {
  assert(!selected('changes', true).includes(emptyPair));
  assert(selected('changes', false).includes(emptyPair));
});
test('all-pair and person-removal inspection keep the zero-candidate example accessible', () => {
  assert(selected('all').includes(emptyPair));
  assert(selected('person').includes(emptyPair));
});
test('default presentation is perspective 360 browsing', () => {
  assert.equal(defaultValue('projection'), 'look');
  assert.match(html, /body data-projection="look"/);
});
test('initial layout has one before panorama and a hidden comparison panel', () => {
  assert.equal([...html.matchAll(/<canvas\b/g)].length, 1);
  assert.match(html, /canvas id="panorama-before"/);
  assert.match(html, /id="ghost-panel" hidden/);
});
test('after-only candidates remain selectable in the before viewer', () => {
  const id = 'R07--D2-f124--D3-f133';
  assert.equal(data.cases.find(c => c.id === id).stats.before.filtered_pixels, 0);
  assert(interaction.cases[id].regions.filtered.simsac.some(r => r.after_pixels > 0));
  assert(selected('changes', true, 'R07').includes(id));
});
test('every default scene has a retained box in the actual exported data', () => {
  for (const id of selected('changes', true, '')) {
    const regions = interaction.cases[id].regions.filtered;
    assert(regions.baseline.length + regions.simsac.length > 0, id);
  }
});
