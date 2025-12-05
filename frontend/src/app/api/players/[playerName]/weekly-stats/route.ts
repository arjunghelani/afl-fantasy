import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ playerName: string }> }
) {
  try {
    const { playerName } = await context.params;
    const searchParams = request.nextUrl.searchParams;
    const url = new URL(`${API_BASE_URL}/players/${encodeURIComponent(playerName)}/weekly-stats`);
    searchParams.forEach((value, key) => {
      url.searchParams.append(key, value);
    });

    const response = await fetch(url.toString(), {
      cache: 'no-store',
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

