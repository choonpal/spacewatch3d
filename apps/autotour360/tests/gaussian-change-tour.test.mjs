import test from 'node:test';
import assert from 'node:assert/strict';
import {changeSource,isTourMessage} from '../gaussian-change-tour.js';

test('change mode defaults to Gaussian while existing data sources remain selectable',()=>{
 assert.equal(changeSource(''), 'gaussian');assert.equal(changeSource('?mode=changes'), 'gaussian');
 assert.equal(changeSource('?source=dorm'),'dorm');assert.equal(changeSource('?source=rescan'),'rescan');
 assert.equal(changeSource('?source=unknown'),'gaussian');
});
test('only messages from the mounted same-origin renderer update the app',()=>{
 const source={},frame={contentWindow:source},origin='http://127.0.0.1:4190';
 const event={origin,source,data:{type:'gaussian-tour:state',events:1}};
 assert.equal(isTourMessage(event,frame,origin),true);
 assert.equal(isTourMessage({...event,origin:'https://elsewhere.test'},frame,origin),false);
 assert.equal(isTourMessage({...event,source:{}},frame,origin),false);
 assert.equal(isTourMessage({...event,data:{type:'different'}},frame,origin),false);
});
