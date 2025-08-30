## Fantasy League Dashboard - Scaffold

This repo will host a dashboard for ESPN Fantasy Football league `86952922` covering seasons 2021, 2022, and 2024.

### Backend (FastAPI)
- Location: `backend/`
- Endpoints:
  - `GET /health` – health check
  - `GET /standings/{year}` – league standings summary for a year (sorted by win %)
  - `GET /playoffs/{year}` – playoff bracket and results for a year

#### Local setup
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open: `http://127.0.0.1:8000/docs` for the interactive API.

### Frontend (Next.js + Tailwind)
- Location: `frontend/`
- Pages:
  - `/` – Standings page with win percentages and team records
  - `/playoffs` – Playoff brackets showing matchups and winners

#### Local setup
```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:3000` for the dashboard.

### Features
- **Standings**: Teams sorted by win percentage with W-L-T records, points for/against
- **Playoff Brackets**: Visual playoff structure with winners highlighted
- **Navigation**: Easy switching between standings and playoffs
- **Responsive**: Works on desktop and mobile

Notes:
- The league is public; no `ESPN_S2`/`SWID` cookies are needed.
- Supported years: 2021, 2022, 2024.
- Standings automatically sort by best record first.
- Playoff data includes fallback structure if ESPN API doesn't provide playoff details.

### Roadmap
- Persistent database for historical snapshots
- CRON/scheduler for periodic refresh
