import test from 'node:test';
import assert from 'node:assert/strict';
import {forecastState,chartSeries,forecastStats,forecastTransition,rangeUndercoverage,targetPayload,viewPayload,matchedObservationCount,isolatedPoints,HOUR} from '../dist/forecast-model.js';
const record={origin:'2026-10-27T10:00:00Z',issuedAt:'2026-10-27T10:20:00Z',
  times:['2026-10-27T11:00:00Z','2026-10-27T12:00:00Z'],areas:{DK1:{solar:[0,100]}}};
test('updated wind uses its own issue, comparison, metrics and ranges without changing solar',()=>{
  const base={...record,areas:{DK1:{wind:[10,20],solar:[0,100]}}};
  const fresh={...base,issuedAt:'2026-10-27T10:40:00Z',areas:{DK1:{wind:[30,40]}}};
  const payload={status:'ready',latest:base,comparison:base,windOutlook:{status:'ready',latest:fresh,comparison:null}};
  assert.equal(targetPayload(payload,'wind'),payload.windOutlook);
  assert.equal(targetPayload(payload,'solar'),payload);
  assert.equal(chartSeries(payload,'DK1','wind').record,fresh);
  assert.equal(chartSeries(payload,'DK1','wind','comparison').record,null);
  assert.equal(chartSeries(payload,'DK1','solar','comparison').record,base);
  assert.ok(chartSeries(payload,'DK1','wind').points.every(p=>p.lower===null));
});
test('failed first wind update retains the original forecast with a delay status',()=>{
  const payload={status:'ready',latest:record,windOutlook:{latest:null,status:'unavailable',message:'provider failed'}};
  assert.equal(targetPayload(payload,'wind').latest,record);
  assert.equal(targetPayload(payload,'wind').status,'delayed');
  assert.equal(payload.status,'ready');
});
test('waiting for the first valid wind issue labels the existing model without inventing a source failure',()=>{
  const payload={status:'ready',latest:record,windOutlook:{latest:null,status:'unavailable',message:null}};
  const selected=targetPayload(payload,'wind');
  assert.equal(selected.latest,record);
  assert.equal(selected.status,'ready');
  assert.equal(selected.windUpdatePending,true);
});
test('a missing current wind comparison offers the older issue only when selected',()=>{
  const previous={...record,areas:{DK1:{wind:[0,20]}}};
  const payload={latest:previous,comparison:previous,windOutlook:{latest:null,comparison:null}};
  const current=viewPayload(payload,'DK1','wind','comparison','current');
  assert.equal(current.comparison,null);
  assert.equal(current.comparisonHasPrevious,true);
  assert.equal(current.comparisonSource,'current');
  assert.equal(chartSeries(payload,'DK1','wind','comparison','previous').record,previous);
  assert.equal(viewPayload(payload,'DK2','wind','comparison','current').comparisonHasPrevious,false);
});
test('range warning needs enough recorded outcomes and a material coverage shortfall',()=>{
  assert.equal(rangeUndercoverage({rangeHours:4000,rangeOrigins:100,rangeCoverage:.52}),true);
  for(const stats of [undefined,{rangeHours:10,rangeOrigins:3,rangeCoverage:0},
    {rangeHours:48,rangeOrigins:1,rangeCoverage:.4},{rangeHours:4000,rangeOrigins:100,rangeCoverage:null},
    {rangeHours:4000,rangeOrigins:100,rangeCoverage:.8}]) assert.equal(rangeUndercoverage(stats),false);
});
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

test('wind comparison keeps usable prior-model history without mixing the outlook or metrics',()=>{
  const times=Array.from({length:48},(_,i)=>new Date(Date.parse('2026-11-15T14:00Z')+i*HOUR).toISOString());
  const old={modelVersion:'old',origin:'2026-11-15T13:00Z',issuedAt:'2026-11-15T13:25Z',times,areas:{DK2:{wind:times.map(()=>200)}}};
  const fresh={...old,modelVersion:'new',origin:'2026-11-16T13:00Z',issuedAt:'2026-11-16T13:25Z',
    times:times.map(t=>new Date(Date.parse(t)+24*HOUR).toISOString())};
  const observations={DK2:times.slice(0,25).map(time=>({time,wind:0}))};
  const payload={latest:old,comparison:old,observations,performance:{old:true},
    windOutlook:{latest:fresh,comparison:fresh,observations,performance:{fresh:true}}};
  const before=structuredClone(payload);
  assert.equal(matchedObservationCount(payload.windOutlook,'DK2','wind'),1);
  let view=viewPayload(payload,'DK2','wind','comparison');
  assert.equal(view.comparisonSource,'previous');assert.equal(view.performance,payload.performance);
  assert.equal(chartSeries(payload,'DK2','wind','comparison').record,old);
  assert.equal(chartSeries(payload,'DK2','wind','comparison','current').record,fresh);
  assert.equal(viewPayload(payload,'DK2','wind','comparison','current').performance,payload.windOutlook.performance);
  assert.equal(chartSeries(payload,'DK2','wind','outlook','previous').record,fresh);
  assert.deepEqual(payload,before);
  payload.windOutlook.observations={DK2:fresh.times.slice(0,24).map(time=>({time,wind:0}))};
  assert.equal(viewPayload(payload,'DK2','wind','comparison').comparisonSource,'current');
  assert.equal(viewPayload(payload,'DK2','wind','comparison','previous').comparisonSource,'previous');
});

test('comparison counts exclude retrospective, missing and not-yet-matured outcomes',()=>{
  const payload={checkedAt:'2026-10-27T12:10Z',comparison:record,
    observations:{DK1:[{time:'2026-10-27T11:00:00+00:00',solar:0},{time:record.times[1],solar:100}]}};
  assert.equal(matchedObservationCount(payload,'DK1','solar'),1);
  payload.comparison={...record,issuedAt:record.times[0]};
  assert.equal(matchedObservationCount(payload,'DK1','solar'),0);
});

test('paired wind paths use the shared issue cutoff, retain zero and keep each range with its model', () => {
  const previous = {...record, areas:{DK1:{wind:[40,120]}},
    ranges:{areas:{DK1:{wind:{lower:[20,80],upper:[50,160]}}}}};
  const current = {...record, issuedAt:record.times[0], areas:{DK1:{wind:[90,0]}}};
  const observations = {DK1:record.times.map(time => ({time,wind:100}))};
  const payload = {latest:previous, observations, windOutlook:{latest:current},
    windComparison:{DK1:{comparison:{previous,current,issuedAfter:current.issuedAt}}}};
  for (const model of ['current','previous']) {
    const {points} = chartSeries(payload,'DK1','wind','comparison',model);
    assert.deepEqual([points[0].observed,points[0].predicted,points[0].alternate,points[0].alternateUpper],
      [100,null,null,null]);
    const {predicted,alternate,upper,alternateUpper} = points[1];
    assert.deepEqual([predicted,alternate,upper,alternateUpper],
      model === 'current' ? [0,120,null,160] : [120,0,160,null]);
  }
  assert.ok(chartSeries(payload,'DK1','wind','outlook').points.every(point => !('alternate' in point)));
});

test('isolated observations remain visible, including zero and points between gaps',()=>{
  const points=[{observed:0},{observed:null},{observed:20},{observed:30},{observed:null},{observed:40}];
  assert.deepEqual(isolatedPoints(points,'observed'),[points[0],points[5]]);
  assert.deepEqual(isolatedPoints([{observed:NaN},{observed:Infinity}],'observed'),[]);
});

test('elapsed targets keep their observations and never contribute to a forecast peak or band', () => {
  const late = {...record, issuedAt: record.times[0], areas: {DK1:{solar:[9999, 40]}},
    ranges: {areas: {DK1:{solar:{lower:[10, 20],upper:[10000, 60]}}}}};
  const series = chartSeries({latest:late,observations:{DK1:[{time:late.times[0],solar:63}]}},'DK1','solar');
  assert.equal(series.points[0].observed,63);
  assert.equal(series.points[0].predicted,null);
  assert.equal(series.points[0].lower,null);
  assert.deepEqual(forecastStats(late,'DK1','solar'),{average:40,hours:1,peak:40,peakTime:late.times[1]});
});
