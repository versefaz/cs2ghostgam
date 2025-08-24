import React, { useEffect, useState } from 'react';
import { gql, useSubscription } from '@apollo/client';

const LIVE_ODDS_SUBSCRIPTION = gql`
  subscription LiveOdds($matchId: ID!) {
    liveOdds(matchId: $matchId) {
      market
      selection
      odds
      movement
      volume
      timestamp
    }
  }
`;

function calculateValueBets(pred: any, liveOdds: any[]) {
  return [];
}

export const LiveAnalytics: React.FC<{ matchId: number }> = ({ matchId }) => {
  const { data } = useSubscription(LIVE_ODDS_SUBSCRIPTION, { variables: { matchId } });
  const [predictions, setPredictions] = useState<any>({ timeline: [], playerImpact: [] });
  const [valueBets, setValueBets] = useState<any[]>([]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/predict/${matchId}`);
        const pred = await res.json();
        setPredictions(pred);
        if (data?.liveOdds && pred) {
          setValueBets(calculateValueBets(pred, data.liveOdds));
        }
      } catch {}
    }, 1000);
    return () => clearInterval(interval);
  }, [matchId, data]);

  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-gray-900 p-4 rounded">
        <h3>Win Probability Timeline</h3>
        {/* integrate recharts later */}
      </div>
      <div className="bg-green-900 p-4 rounded">
        <h3>💰 Value Bets (Realtime)</h3>
        {valueBets.map((bet, i) => (
          <div key={i} className="border-b py-2">
            <div className="flex justify-between">
              <span>{bet.market}: {bet.selection}</span>
              <span className="text-green-400">EV: {(bet.ev*100).toFixed(2)}%</span>
            </div>
            <div className="text-sm text-gray-400">
              Odds: {bet.odds} | Edge: {(bet.edge*100).toFixed(2)}%
            </div>
          </div>
        ))}
      </div>
      <div className="bg-blue-900 p-4 rounded">
        <h3>Player Impact</h3>
      </div>
    </div>
  );
};
