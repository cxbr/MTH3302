import fs from 'fs';
import { getHistoricalRates, instrumentMetaData } from 'dukascopy-node';
import * as lib from 'dukascopy-node';

const out = new URL('./hot20_duka_probe_output/', import.meta.url);
fs.mkdirSync(out, { recursive: true });
const exported = Object.keys(lib).sort();
const metaType = Array.isArray(instrumentMetaData) ? 'array' : typeof instrumentMetaData;
const metaKeys = instrumentMetaData && typeof instrumentMetaData === 'object' ? Object.keys(instrumentMetaData).slice(0,50) : [];
const metaSample = Array.isArray(instrumentMetaData)
  ? instrumentMetaData.slice(0,10)
  : metaKeys.slice(0,10).map((key) => [key, instrumentMetaData[key]]);
const metaText = JSON.stringify(instrumentMetaData);
const sought = ['aaplususd','msftususd','nvdaususd','amznususd','metaususd','fbususd','googlususd','tslaususd','amdususd','jpmususd'];
const found = Object.fromEntries(sought.map((id) => [id, metaText.includes(id)]));
console.log('EXPORTS', exported);
console.log('META', metaType, metaKeys.length, JSON.stringify(metaSample).slice(0,5000), found);
const data = await getHistoricalRates({
  instrument: 'aaplususd',
  dates: { from: new Date('2024-01-02T00:00:00Z'), to: new Date('2024-01-04T00:00:00Z') },
  timeframe: 'm15',
  format: 'json',
  volumes: true,
});
console.log('TYPE', typeof data, Array.isArray(data), data?.length);
console.log('SAMPLE', JSON.stringify(Array.isArray(data) ? data.slice(0,5) : data, null, 2).slice(0,5000));
fs.writeFileSync(new URL('probe.json', out), JSON.stringify({
  exported,
  metaType,
  metaKeys,
  metaSample,
  found,
  type: typeof data,
  isArray: Array.isArray(data),
  length: data?.length,
  sample: Array.isArray(data) ? data.slice(0,20) : data,
}, null, 2));
