import fs from 'fs';
import path from 'path';
import dukascopy from 'dukascopy-node';
const { getHistoricalRates } = dukascopy;

const OUT = path.resolve('cross_asset_v9_data');
fs.mkdirSync(OUT, { recursive: true });

const instruments = [
  ['SPY_ETF', 'spyususd', 'ETF'],
  ['QQQ_ETF', 'qqqususd', 'ETF'],
  ['SP500_INDEX', 'usa500idxusd', 'Index CFD'],
  ['DAX_INDEX', 'deuidxeur', 'Index CFD'],
  ['NIKKEI_INDEX', 'jpnidxjpy', 'Index CFD'],
  ['EURUSD_FX', 'eurusd', 'FX'],
  ['USDJPY_FX', 'usdjpy', 'FX'],
  ['GBPUSD_FX', 'gbpusd', 'FX'],
  ['GOLD_SPOT', 'xauusd', 'Metal'],
  ['BRENT_CFD', 'brentcmdusd', 'Energy CFD'],
  ['BTCUSD_CRYPTO', 'btcusd', 'Crypto'],
];

const jobs = [];
for (const [alias, instrument, category] of instruments) {
  for (const priceType of ['bid', 'ask']) jobs.push({ alias, instrument, category, priceType });
}

const manifest = [];
const CONCURRENCY = 4;
let cursor = 0;

function summarizeUnexpected(data) {
  if (data === null) return 'null';
  if (Array.isArray(data)) return `array(${data.length})`;
  if (typeof data === 'object') {
    try { return `object:${JSON.stringify(data).slice(0, 500)}`; }
    catch { return `object:${Object.prototype.toString.call(data)}`; }
  }
  return `${typeof data}:${String(data).slice(0, 500)}`;
}

async function downloadOne(job) {
  const { alias, instrument, category, priceType } = job;
  const file = path.join(OUT, `${alias}_${priceType}.csv`);
  if (fs.existsSync(file) && fs.statSync(file).size > 1000) {
    const bytes = fs.statSync(file).size;
    console.log(`CACHE ${alias} ${priceType}: ${bytes} bytes`);
    manifest.push({ ...job, file, bytes, cached: true, status: 'ok' });
    return;
  }
  console.log(`DOWNLOAD ${alias} ${instrument} ${priceType}`);
  try {
    const data = await getHistoricalRates({
      instrument,
      dates: { from: '2020-01-01', to: '2026-01-01' },
      timeframe: 'm15',
      priceType,
      format: 'csv',
      utcOffset: 0,
      volumes: true,
      volumeUnits: 'units',
      ignoreFlats: true,
      batchSize: 30,
      pauseBetweenBatchesMs: 100,
      useCache: true,
      cacheFolderPath: path.resolve('.dukascopy-cache'),
      retryCount: 4,
      retryOnEmpty: true,
      failAfterRetryCount: false,
      pauseBetweenRetriesMs: 750,
    });
    if (typeof data !== 'string' || !data.startsWith('timestamp,')) {
      const error = `Unexpected output: ${summarizeUnexpected(data)}`;
      console.warn(`SKIP ${alias} ${priceType}: ${error}`);
      manifest.push({ ...job, file, bytes: 0, cached: false, status: 'error', error });
      return;
    }
    fs.writeFileSync(file, data);
    const bytes = fs.statSync(file).size;
    console.log(`SAVED ${alias} ${priceType}: ${bytes} bytes`);
    manifest.push({ ...job, file, bytes, cached: false, status: 'ok' });
  } catch (err) {
    const error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    console.warn(`SKIP ${alias} ${priceType}: ${error}`);
    manifest.push({ ...job, file, bytes: 0, cached: false, status: 'error', error });
  }
}

async function worker(workerId) {
  while (true) {
    const index = cursor++;
    if (index >= jobs.length) return;
    console.log(`WORKER ${workerId}: job ${index + 1}/${jobs.length}`);
    await downloadOne(jobs[index]);
  }
}

await Promise.all(Array.from({ length: CONCURRENCY }, (_, i) => worker(i + 1)));
manifest.sort((a, b) => `${a.alias}_${a.priceType}`.localeCompare(`${b.alias}_${b.priceType}`));
fs.writeFileSync(path.join(OUT, 'download_manifest.json'), JSON.stringify(manifest, null, 2));

const successfulAliases = instruments
  .map(([alias]) => alias)
  .filter(alias => ['bid', 'ask'].every(side => manifest.some(x => x.alias === alias && x.priceType === side && x.status === 'ok')));
console.log(`Completed ${manifest.filter(x => x.status === 'ok').length}/${manifest.length} side downloads; ${successfulAliases.length} complete bid/ask markets`);
console.log(`Complete markets: ${successfulAliases.join(', ')}`);
if (successfulAliases.length < 9) throw new Error(`Only ${successfulAliases.length} complete bid/ask markets; need at least 9`);
