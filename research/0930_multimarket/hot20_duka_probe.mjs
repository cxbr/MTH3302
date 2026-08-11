import fs from 'fs';
import { getHistoricalRates } from 'dukascopy-node';
import * as lib from 'dukascopy-node';

const out = new URL('./hot20_duka_probe_output/', import.meta.url);
fs.mkdirSync(out, { recursive: true });
const exported = Object.keys(lib).sort();
console.log('EXPORTS', exported);
const data = await getHistoricalRates({
  instrument: 'aaplususd',
  dates: { from: new Date('2024-01-02T00:00:00Z'), to: new Date('2024-01-04T00:00:00Z') },
  timeframe: 'm15',
  format: 'json',
  volumes: true,
});
console.log('TYPE', typeof data, Array.isArray(data), data?.length);
console.log('SAMPLE', JSON.stringify(Array.isArray(data) ? data.slice(0,5) : data, null, 2).slice(0,5000));
fs.writeFileSync(new URL('probe.json', out), JSON.stringify({ exported, type: typeof data, isArray: Array.isArray(data), length: data?.length, sample: Array.isArray(data) ? data.slice(0,20) : data }, null, 2));
