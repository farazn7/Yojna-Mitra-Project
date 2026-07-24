// Vercel serverless function — proxies Healthchecks.io status so the API key
// never reaches the browser. Matches checks to bots by their ping URL UUID.

const TELEGRAM_UUID = '6f985ce4-0f9d-40ab-9b0f-d3dd68cd2957';
const DISCORD_UUID  = 'c38c6455-a777-4a12-b524-8b8dec9ef57d';

export default async function handler(req, res) {
  const apiKey = process.env.HEALTHCHECKS_API;

  if (!apiKey) {
    return res.status(500).json({ error: 'HEALTHCHECKS_API not configured' });
  }

  try {
    const response = await fetch('https://healthchecks.io/api/v3/checks/', {
      headers: { 'X-Api-Key': apiKey },
    });

    if (!response.ok) {
      throw new Error(`Healthchecks returned ${response.status}`);
    }

    const { checks } = await response.json();

    // Read-only API keys obscure the ping_url for security, so we match by check name
    let telegramStatus = 'unknown';
    let discordStatus  = 'unknown';

    for (const check of checks) {
      if (check.name === 'Telegram Bot') {
        telegramStatus = check.status; // 'up', 'down', 'new', 'grace', 'paused'
      } else if (check.name === 'Discord Bot') {
        discordStatus = check.status;
      }
    }

    res.setHeader('Cache-Control', 's-maxage=30, stale-while-revalidate=60');
    return res.status(200).json({ telegram: telegramStatus, discord: discordStatus });

  } catch (err) {
    console.error('[status] Healthchecks fetch failed:', err.message);
    return res.status(502).json({ error: 'Failed to reach Healthchecks.io' });
  }
}
