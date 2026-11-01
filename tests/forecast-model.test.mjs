import test from 'node:test';
import assert from 'node:assert/strict';
import {forecastState,chartSeries,forecastStats,forecastTransition,HOUR} from '../dist/forecast-model.js';
const record={origin:'2026-10-27T10:00:00Z',issuedAt:'2026-10-27T10:20:00Z',
  times:['2026-10-27T11:00:00Z','2026-10-27T12:00:00Z'],areas:{DK1:{solar:[0,100]}}};
test('old forecasts are visibly delayed even if the last job said ready',()=>{
  const p={status:'ready',latest:record};
  assert.equal(forecastState(p,Date.parse(record.issuedAt)+3*HOUR),'delayed');
  assert.equal(forecastState(p,Date.parse(record.issuedAt)+HOUR),'ready');
  assert.equal(forecastState({status:'ready',latest:null}),'unavailable');
});
test('no comparison is invented before a forecast has aged',()=>{
  assert.deepEqual(chartSeries({latest:record},'DK1','solar','comparison'),{record:null,points:[]});
});
test('zero solar generation is preserved and missing actuals remain missing',()=>{
  const p={latest:record,observations:{DK1:[{time:'2026-10-27T11:00:00Z',solar:0}]}};
  const s=chartSeries(p,'DK1','solar');
  assert.equal(s.points[0].observed,0);assert.equal(s.points[0].predicted,0);
  assert.equal(s.points[1].observed,null);assert.equal(forecastStats(record,'DK1','solar').average,50);
});
test('ranges remain attached to their original forecast and missing nights stay missing',()=>{
  const ranged={...record,ranges:{areas:{DK1:{solar:{lower:[null,80],upper:[null,140]}}}}};
  const p={latest:ranged,comparison:record};
  const points=chartSeries(p,'DK1','solar').points;
  assert.equal(points[0].lower,null);assert.equal(points[1].lower,80);assert.equal(points[1].upper,140);
  assert.ok(chartSeries(p,'DK1','solar','comparison').points.every(p=>p.lower===null&&p.upper===null));
});

test('transition links adjacent real values, including zero, without inventing predictions',()=>{
  const payload={latest:record,observations:{DK1:[{time:record.origin,solar:0}]}};
  const before=structuredClone(payload);
  const {points}=chartSeries(payload,'DK1','solar');
  const originalPoints=structuredClone(points);
  assert.deepEqual(forecastTransition(points,record.origin),[
    {time:Date.parse(record.origin),value:0},
    {time:Date.parse(record.times[0]),value:0},
  ]);
  assert.deepEqual(payload,before);
  assert.deepEqual(points,originalPoints);
  assert.equal(points[0].predicted,null);
  assert.equal(points[0].lower,null);
});

test('transition never bridges missing or invalid boundary values',()=>{
  const start=Date.parse(record.origin);
  const complete=[{time:start,observed:20,predicted:null},{time:start+HOUR,observed:null,predicted:30}];
  for(const value of [null,undefined,NaN,Infinity]){
    assert.equal(forecastTransition([{...complete[0],observed:value},complete[1]],record.origin),null);
    assert.equal(forecastTransition([complete[0],{...complete[1],predicted:value}],record.origin),null);
  }
  assert.equal(forecastTransition([complete[1]],record.origin),null);
  assert.equal(forecastTransition([complete[0],{...complete[1],time:start+2*HOUR}],record.origin),null);
  assert.equal(forecastTransition([{...complete[0],time:start-HOUR},complete[1]],record.origin),null);
});

test('transition stays absent when observations overlap forecasts or the view lacks history',()=>{
  const start=Date.parse(record.origin);
  assert.equal(forecastTransition([
    {time:start,observed:20,predicted:null},
    {time:start+HOUR,observed:0,predicted:30},
  ],record.origin),null);
  assert.equal(forecastTransition([
    {time:start,observed:20,predicted:22},
    {time:start+HOUR,observed:null,predicted:30},
  ],record.origin),null);
  const points=chartSeries({comparison:record},'DK1','solar','comparison').points;
  assert.equal(forecastTransition(points,record.origin),null);
  assert.equal(forecastTransition([],record.origin),null);
});

test('transition adjacency uses UTC hours across midnight, year changes and DST',()=>{
  for(const origin of ['2026-11-01T23:00:00Z','2026-12-31T23:00:00Z','2026-10-25T02:00:00+02:00']){
    const start=Date.parse(origin);
    const points=[{time:start,observed:20,predicted:null},{time:start+HOUR,observed:null,predicted:30}];
    assert.deepEqual(forecastTransition(points,origin),[{time:start,value:20},{time:start+HOUR,value:30}]);
  }
});
