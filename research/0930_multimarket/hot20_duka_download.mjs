import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { fileURLToPath } from 'url';
import { getHistoricalRates, instrumentMetaData } from 'dukascopy-node';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, 'hot20_v6_work', 'raw');
fs.mkdirSync(OUT, { recursive: true });

const TICKERS = [
  'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','AMD','AVGO','NFLX','INTC','QCOM','MU','AMAT','LRCX','KLAC',
  'CRM','ORCL','ADBE','CSCO','IBM','NOW','PANW','CRWD','PLTR','SNOW','SHOP','PYPL','SQ','COIN','MSTR','MARA','RIOT',
  'UBER','ABNB','DASH','RBLX','RIVN','LCID','F','GM','BA','CAT','DE','GE','HON','LMT','RTX','NOC','GD','DAL','AAL',
  'UAL','UPS','FDX','XOM','CVX','COP','OXY','SLB','HAL','JPM','BAC','C','WFC','GS','MS','SCHW','BLK','AXP','V','MA',
  'WMT','COST','TGT','HD','LOW','NKE','SBUX','MCD','DIS','CMCSA','T','VZ','TMUS','KO','PEP','PG','CL','KHC','PM','MO',
  'JNJ','PFE','MRK','ABBV','LLY','UNH','CVS','CI','HUM','TMO','DHR','ABT','MDT','BMY','AMGN','GILD','ISRG','REGN','VRTX',
  'ZM','ROKU','SNAP','PINS','SPOT','DOCU','TWLO','DDOG','NET','MDB','TEAM','ZS','OKTA','DKNG','HOOD','AFRM','SOFI',
  'ENPH','FSLR','NEE','DUK','SO','AEP','EXC','PLUG','FCEL','BABA','JD','PDD','NIO','TSM','ARKK','QQQ','SPY','IWM'
];

const ALIASES = {
  META: ['META', 'FB'],
  SQ: ['SQ', 'XYZ'],
  GOOGL: ['GOOGL'],
  GOOG: ['GOOG'],
};

function normTicker(text) {
  return String(text || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
}

function findInstrument(ticker) {
  const aliases = ALIASES[ticker] || [ticker];
  const targets = new Set(aliases.map(normTicker));
  const candidates = [];
  for (const [id, meta] of Object.entries(instrumentMetaData)) {
    const name = String(meta?.name || '').toUpperCase();
    const code = String(meta?.code || '').toUpperCase();
    if (!(name.includes('.US/USD') || code.includes('.US-USD'))) continue;
    const symbolName = normTicker(name.split('.')[0]);
    const symbolCode = normTicker(code.split('.')[0]);
    if (targets.has(symbolName) || targets.has(symbolCode)) {
      candidates.push({ id, meta });
    }
  }
  if (!candidates.length) return null;
  candidates.sort((a, b) => {
    const aStart = Date.parse(a.meta?.startDayForMinuteCandles || '2999-01-01');
    const bStart = Date.parse(b.meta?.startDayForMinuteCandles || '2999-01-01');
    return aStart - bStart;
  });
  return candidates[0];
}

function csvText(rows) {
  const lines = ['timestamp,open,high,low,close,volume'];
  for (const row of rows) {
    lines.push([row.timestamp, row.open, row.high, row.low, row.close, row.volume ?? 0].join(','));
  }
  return lines.join('\n') + '\n';
}

async function downloadTicker(ticker) {
  const found = findInstrument(ticker);
  if (!found) return { ticker, status: 'missing_metadata' };
  const startMeta = new Date(found.meta.startDayForMinuteCandles || '2018-01-01T00:00:00Z');
  const from = new Date(Math.max(startMeta.getTime(), Date.parse('2018-01-01T00:00:00Z')));
  const to = new Date('2026-01-01T00:00:00Z');
  try {
    const rows = await getHistoricalRates({
      instrument: found.id,
      dates: { from, to },
      timeframe: 'm15',
      format: 'json',
      volumes: true,
    });
    if (!Array.isArray(rows) || rows.length < 5000) {
      return { ticker, instrument: found.id, status: 'insufficient_rows', rows: Array.isArray(rows) ? rows.length : null };
    }
    const file = path.join(OUT, `${ticker}.csv.gz`);
    fs.writeFileSync(file, zlib.gzipSync(csvText(rows), { level: 6 }));
    return {
      ticker,
      instrument: found.id,
      name: found.meta.name,
      code: found.meta.code,
      status: 'downloaded',
      rows: rows.length,
      start: new Date(rows[0].timestamp).toISOString(),
      end: new Date(rows[rows.length - 1].timestamp).toISOString(),
      bytes: fs.statSync(file).size,
      source_start: found.meta.startDayForMinuteCandles,
    };
  } catch (error) {
    return { ticker, instrument: found.id, status: 'error', error: String(error?.stack || error) };
  }
}

async function runPool(items, concurrency) {
  const results = new Array(items.length);
  let cursor = 0;
  async function worker(workerId) {
    while (true) {
      const index = cursor++;
      if (index >= items.length) return;
      const ticker = items[index];
      console.log(`[${index + 1}/${items.length}] worker=${workerId} ${ticker}`);
      const result = await downloadTicker(ticker);
      results[index] = result;
      console.log(JSON.stringify(result));
    }
  }
  await Promise.all(Array.from({ length: concurrency }, (_, i) => worker(i + 1)));
  return results;
}

const results = await runPool(TICKERS, 10);
const manifest = {
  requested: TICKERS.length,
  downloaded: results.filter((r) => r.status === 'downloaded').length,
  generated_at: new Date().toISOString(),
  concurrency: 10,
  results,
};
fs.writeFileSync(path.join(HERE, 'hot20_v6_work', 'download_manifest.json'), JSON.stringify(manifest, null, 2));
console.log(JSON.stringify({ requested: manifest.requested, downloaded: manifest.downloaded, concurrency: manifest.concurrency }, null, 2));
if (manifest.downloaded < 40) {
  throw new Error(`Only ${manifest.downloaded} tickers downloaded; need at least 40 for a credible top-20 scanner.`);
}
