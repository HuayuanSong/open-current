import test from 'node:test';
import assert from 'node:assert/strict';
import {forecastState,chartSeries,forecastStats,HOUR} from '../dist/forecast-model.js';
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
