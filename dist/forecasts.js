import {forecastState,chartSeries,forecastStats,usable,HOUR} from "./forecast-model.js?v=71d9f264a613";

const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const d3=window.d3;
const state={area:"DK1",target:"wind",view:"outlook"};
const names={wind:"Wind generation",solar:"Solar generation",load:"Electricity demand"};
let payload=null, activePoints=[],activeIndex=0;
const utc=value=>new Intl.DateTimeFormat("en-GB",{day:"numeric",month:"short",hour:"2-digit",minute:"2-digit",timeZone:"UTC"}).format(new Date(value))+" UTC";
const power=value=>usable(value)?(value>=1000?`${(value/1000).toFixed(2)} GW`:`${Math.round(value)} MW`):"—";
const text=(selector,value)=>{$(selector).textContent=value;};

function stat(label,value,note){
  const el=document.createElement("div");
  for(const [tag,cls,content] of [["span","metric-label",label],["strong","",value],["span","muted",note]]){
    const child=document.createElement(tag);child.className=cls;child.textContent=content;el.append(child);
  }
  return el;
}

function render(){
  const status=forecastState(payload);
  $("#forecast-status").dataset.status=status;
  if(!payload?.latest){
    text("#forecast-status","The next forecast is waiting for complete source data.");
  }else{
    const missing=payload.latest.quality?.[state.area]?.filledInputHours || 0;
    text("#forecast-status",`${status==="delayed"?"Update delayed · Last forecast":"Forecast issued"} ${utc(payload.latest.issuedAt)}${missing?` · ${missing} missing input hours filled from earlier readings`:""}`);
  }
  $$('[data-forecast-area]').forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.forecastArea===state.area)));
  $$('[data-forecast-target]').forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.forecastTarget===state.target)));
  $$('[data-forecast-view]').forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.forecastView===state.view)));
  const {record,points}=chartSeries(payload,state.area,state.target,state.view);
  $("#forecast-empty").hidden=Boolean(record);
  text("#forecast-empty",state.view==="comparison"?"We’re saving forecasts now. Comparisons will appear as the next hourly observations arrive.":"A forecast will appear when electricity and weather data are available.");
  const stats=forecastStats(record,state.area,state.target), live=payload?.performance?.[state.area]?.[state.target];
  $("#forecast-stats").replaceChildren(...(stats?[
    stat("48-hour average",power(stats.average),names[state.target]),
    stat("Forecast peak",power(stats.peak),utc(stats.peakTime)),
    stat("Live mean error",live?.forecastHours>=24?power(live.maeMW):"Collecting outcomes",live?.forecastHours?`${live.forecastHours} forecasted hours · ${live.origins} issued outlooks · last 7 days`:"Measured after forecasts are issued")
  ]:[]));
  text("#forecast-timing",record?`${state.view==="comparison"?`Comparing the forecast issued ${utc(record.issuedAt)}. `:""}Hourly average power · all times UTC. Weather run: ${utc(record.weatherRun)}.`:"Hourly average power · all times UTC.");
  $("#forecast-download").disabled=!record;
  draw(record,points);
}

function draw(record,points){
  const svg=d3.select("#forecast-chart");svg.selectAll("*").remove();$("#forecast-tooltip").hidden=true;activePoints=points;
  const width=$("#forecast-chart-wrap").clientWidth,height=width<600?300:360;
  svg.attr("viewBox",`0 0 ${width} ${height}`);
  if(!record||!points.length)return;
  svg.append("title").text(`${names[state.target]} · ${state.area} · hourly power in gigawatts`);
  svg.append("desc").text("Solid line shows recorded readings. Dashed green line shows predictions. Use arrow keys to inspect each hour.");
  const margin={top:24,right:24,bottom:48,left:58};
  const x=d3.scaleUtc().domain(d3.extent(points,d=>new Date(d.time))).range([margin.left,width-margin.right]);
  const max=d3.max(points,d=>Math.max(d.observed??0,d.predicted??0))||1;
  const y=d3.scaleLinear().domain([0,max/1000*1.08]).nice().range([height-margin.bottom,margin.top]);
  const axis=svg.append("g").attr("class","forecast-axis");
  axis.append("g").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(width-margin.left-margin.right)).tickFormat(d=>`${d}`));
  axis.append("g").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(width<600?4:8).tickFormat(d3.utcFormat("%d %b %Hh")));
  svg.append("text").attr("x",margin.left).attr("y",15).attr("class","forecast-unit").text("GW");
  const origin=Date.parse(record.origin);
  if(origin>points[0].time&&origin<points.at(-1).time){
    svg.append("line").attr("x1",x(origin)).attr("x2",x(origin)).attr("y1",margin.top).attr("y2",height-margin.bottom).attr("class","forecast-origin");
    svg.append("text").attr("x",x(origin)+8).attr("y",margin.top+13).attr("class","forecast-unit").text("Outlook →");
  }
  for(const key of ['observed','predicted']){
    const line=d3.line().defined(d=>usable(d[key])).x(d=>x(d.time)).y(d=>y(d[key]/1000));
    svg.append("path").datum(points).attr("d",line).attr("class",`forecast-line forecast-${key}`);
  }
  const cursor=svg.append("line").attr("class","forecast-cursor").attr("y1",margin.top).attr("y2",height-margin.bottom).style("display","none");
  function inspect(index){
    activeIndex=Math.max(0,Math.min(points.length-1,index));const p=points[activeIndex];
    cursor.attr("x1",x(p.time)).attr("x2",x(p.time)).style("display",null);
    const tip=$("#forecast-tooltip");tip.replaceChildren();
    for(const t of [utc(p.time),`Observed: ${power(p.observed)}`,`Forecast: ${power(p.predicted)}`]){const row=document.createElement("div");row.textContent=t;tip.append(row);}
    tip.hidden=false;tip.style.left=`${Math.max(8,Math.min(width-230,x(p.time)+12))}px`;
  }
  svg.on("pointermove",event=>{const [px]=d3.pointer(event);const stamp=+x.invert(px);inspect(d3.bisector(d=>d.time).center(points,stamp));});
  svg.on("pointerleave",()=>{cursor.style("display","none");$("#forecast-tooltip").hidden=true;});
  svg.on("keydown",event=>{if(event.key==="ArrowRight"||event.key==="ArrowLeft"){event.preventDefault();inspect(activeIndex+(event.key==="ArrowRight"?1:-1));}if(event.key==="Escape"){$("#forecast-tooltip").hidden=true;cursor.style("display","none");}});
}

for(const [attr,key] of [["area","area"],["target","target"],["view","view"]]){
  $$(`[data-forecast-${attr}]`).forEach(b=>b.addEventListener("click",()=>{state[key]=b.dataset[`forecast${attr[0].toUpperCase()+attr.slice(1)}`];render();}));
}
$("#forecast-download").onclick=()=>{
  const record=state.view==="comparison"?payload.comparison:payload.latest;if(!record)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:"application/json"}));
  const a=document.createElement("a");a.href=url;a.download=`open-current-forecast-${record.origin.slice(0,10)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
async function refresh(){
  if(document.hidden)return;
  try{const response=await fetch("data/forecasts.json",{cache:"no-cache"});if(!response.ok)throw new Error(response.status);const next=await response.json();if(next.schema!==1)throw new Error("schema");payload=next;render();}
  catch{if(payload)render();else{text("#forecast-status","Forecasts are temporarily unavailable. Please try again shortly.");$("#forecast-status").dataset.status="unavailable";}}
}
let resize;
new ResizeObserver(()=>{clearTimeout(resize);resize=setTimeout(render,80);}).observe($("#forecast-chart-wrap"));
refresh();setInterval(refresh,5*60*1000);document.addEventListener("visibilitychange",()=>{if(!document.hidden)refresh();});
