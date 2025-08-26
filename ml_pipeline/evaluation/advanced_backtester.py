import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class BacktestEvent:
    timestamp: datetime
    event_type: str  # 'odds_update', 'match_start', 'match_end', 'prediction'
    match_id: str
    data: Dict[str, Any]


@dataclass
class BacktestPosition:
    position_id: str
    match_id: str
    team: str
    market_type: str
    stake: float
    odds: float
    timestamp: datetime
    prediction_confidence: float
    expected_value: float
    result: Optional[str] = None
    pnl: Optional[float] = None
    close_timestamp: Optional[datetime] = None


@dataclass
class BacktestMetrics:
    total_bets: int
    winning_bets: int
    losing_bets: int
    void_bets: int
    hit_rate: float
    total_staked: float
    total_return: float
    net_profit: float
    roi: float
    max_drawdown: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    turnover: float
    avg_bet_size: float
    largest_win: float
    largest_loss: float
    longest_winning_streak: int
    longest_losing_streak: int
    current_streak: int
    current_streak_type: str
    profit_by_month: Dict[str, float]
    hit_rate_by_month: Dict[str, float]
    performance_by_market: Dict[str, Dict[str, float]]
    performance_by_confidence: Dict[str, Dict[str, float]]


class EventDrivenBacktester:
    """Advanced event-driven backtester with comprehensive metrics"""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.positions: Dict[str, BacktestPosition] = {}
        self.closed_positions: List[BacktestPosition] = []
        self.events: List[BacktestEvent] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdown_curve: List[Tuple[datetime, float]] = []
        
        # Risk management
        self.max_bet_size = 0.05
        self.max_exposure = 0.20
        self.current_exposure = 0.0
        
        # Performance tracking
        self.peak_equity = initial_bankroll
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_dd_start: Optional[datetime] = None
        self.max_dd_end: Optional[datetime] = None
        
        self.db_path = "backtest_results.db"
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                start_date TEXT,
                end_date TEXT,
                initial_bankroll REAL,
                final_bankroll REAL,
                total_bets INTEGER,
                hit_rate REAL,
                roi REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                created_at TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_positions (
                position_id TEXT PRIMARY KEY,
                run_id TEXT,
                match_id TEXT,
                team TEXT,
                market_type TEXT,
                stake REAL,
                odds REAL,
                timestamp TEXT,
                prediction_confidence REAL,
                expected_value REAL,
                result TEXT,
                pnl REAL,
                close_timestamp TEXT
            )
        """)
        conn.close()
    
    async def load_historical_data(self, data_path: str) -> List[BacktestEvent]:
        events = []
        
        if Path(data_path).suffix == '.json':
            with open(data_path, 'r') as f:
                raw_data = json.load(f)
                
            for item in raw_data:
                event = BacktestEvent(
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    event_type=item['event_type'],
                    match_id=item['match_id'],
                    data=item['data']
                )
                events.append(event)
        
        elif Path(data_path).suffix == '.csv':
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            for _, row in df.iterrows():
                event = BacktestEvent(
                    timestamp=row['timestamp'],
                    event_type=row['event_type'],
                    match_id=row['match_id'],
                    data=row.to_dict()
                )
                events.append(event)
        
        events.sort(key=lambda x: x.timestamp)
        return events
    
    async def run_backtest(self, 
                          events: List[BacktestEvent],
                          prediction_model,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> BacktestMetrics:
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        logger.info(f"Running backtest on {len(events)} events")
        
        for event in events:
            await self._process_event(event, prediction_model)
            
            self.equity_curve.append((event.timestamp, self.current_bankroll))
            
            if self.current_bankroll > self.peak_equity:
                self.peak_equity = self.current_bankroll
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_equity - self.current_bankroll) / self.peak_equity
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
                    if not self.max_dd_start:
                        self.max_dd_start = event.timestamp
                    self.max_dd_end = event.timestamp
            
            self.drawdown_curve.append((event.timestamp, self.current_drawdown))
        
        metrics = self._calculate_metrics()
        await self._save_results(metrics)
        return metrics
    
    async def _process_event(self, event: BacktestEvent, prediction_model):
        if event.event_type == 'odds_update':
            await self._handle_odds_update(event, prediction_model)
        elif event.event_type == 'match_end':
            await self._handle_match_end(event)
    
    async def _handle_odds_update(self, event: BacktestEvent, prediction_model):
        match_data = event.data
        
        if self._should_bet(match_data):
            prediction = await prediction_model.predict(match_data)
            
            if prediction and prediction.get('expected_value', 0) > 0.05:
                await self._place_bet(event, prediction)
    
    async def _handle_match_end(self, event: BacktestEvent):
        match_id = event.match_id
        result_data = event.data
        
        positions_to_close = [
            (pos_id, pos) for pos_id, pos in self.positions.items() 
            if pos.match_id == match_id
        ]
        
        for pos_id, position in positions_to_close:
            result = self._determine_position_result(position, result_data)
            pnl = self._calculate_pnl(position, result)
            
            position.result = result
            position.pnl = pnl
            position.close_timestamp = event.timestamp
            
            self.closed_positions.append(position)
            del self.positions[pos_id]
            
            self.current_bankroll += pnl
            self.current_exposure -= position.stake
    
    def _should_bet(self, match_data: Dict) -> bool:
        if self.current_exposure >= self.max_exposure * self.current_bankroll:
            return False
        
        if not match_data.get('odds_team1') or not match_data.get('odds_team2'):
            return False
        
        return True
    
    async def _place_bet(self, event: BacktestEvent, prediction: Dict):
        confidence = prediction.get('confidence', 0.5)
        odds = prediction.get('odds', 2.0)
        expected_value = prediction.get('expected_value', 0)
        
        if expected_value <= 0:
            return
        
        kelly_fraction = (confidence * odds - 1) / (odds - 1)
        kelly_stake = self.current_bankroll * kelly_fraction * 0.25
        
        max_stake = self.current_bankroll * self.max_bet_size
        stake = min(kelly_stake, max_stake)
        
        if stake < 10:
            return
        
        position = BacktestPosition(
            position_id=f"pos_{event.match_id}_{len(self.closed_positions)}",
            match_id=event.match_id,
            team=prediction.get('team', 'unknown'),
            market_type=prediction.get('market_type', 'match_winner'),
            stake=stake,
            odds=odds,
            timestamp=event.timestamp,
            prediction_confidence=confidence,
            expected_value=expected_value
        )
        
        self.positions[position.position_id] = position
        self.current_exposure += stake
    
    def _determine_position_result(self, position: BacktestPosition, result_data: Dict) -> str:
        winner = result_data.get('winner')
        if not winner:
            return 'void'
        
        if position.market_type == 'match_winner':
            return 'win' if position.team.lower() == winner.lower() else 'loss'
        
        return 'void'
    
    def _calculate_pnl(self, position: BacktestPosition, result: str) -> float:
        if result == 'win':
            return position.stake * (position.odds - 1)
        elif result == 'loss':
            return -position.stake
        else:
            return 0.0
    
    def _calculate_metrics(self) -> BacktestMetrics:
        if not self.closed_positions:
            return BacktestMetrics(
                total_bets=0, winning_bets=0, losing_bets=0, void_bets=0,
                hit_rate=0.0, total_staked=0.0, total_return=0.0,
                net_profit=0.0, roi=0.0, max_drawdown=0.0,
                max_drawdown_duration=0, sharpe_ratio=0.0,
                sortino_ratio=0.0, calmar_ratio=0.0, turnover=0.0,
                avg_bet_size=0.0, largest_win=0.0, largest_loss=0.0,
                longest_winning_streak=0, longest_losing_streak=0,
                current_streak=0, current_streak_type='none',
                profit_by_month={}, hit_rate_by_month={},
                performance_by_market={}, performance_by_confidence={}
            )
        
        total_bets = len(self.closed_positions)
        winning_bets = len([p for p in self.closed_positions if p.result == 'win'])
        losing_bets = len([p for p in self.closed_positions if p.result == 'loss'])
        void_bets = len([p for p in self.closed_positions if p.result == 'void'])
        
        total_staked = sum(p.stake for p in self.closed_positions)
        total_return = sum(p.pnl for p in self.closed_positions if p.pnl)
        net_profit = total_return
        roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0
        
        hit_rate = (winning_bets / (winning_bets + losing_bets)) * 100 if (winning_bets + losing_bets) > 0 else 0
        
        returns = [p.pnl / p.stake for p in self.closed_positions if p.pnl is not None and p.stake > 0]
        
        if returns:
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            risk_free_rate = 0.02 / 252
            sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
            
            downside_returns = returns_array[returns_array < mean_return]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
            sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            calmar_ratio = (roi / 100) / self.max_drawdown if self.max_drawdown > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
        
        max_dd_duration = 0
        if self.max_dd_start and self.max_dd_end:
            max_dd_duration = (self.max_dd_end - self.max_dd_start).days
        
        avg_bet_size = total_staked / total_bets if total_bets > 0 else 0
        pnls = [p.pnl for p in self.closed_positions if p.pnl is not None]
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0
        
        streaks = self._calculate_streaks()
        profit_by_month = self._calculate_monthly_performance()
        hit_rate_by_month = self._calculate_monthly_hit_rates()
        performance_by_market = self._calculate_market_performance()
        performance_by_confidence = self._calculate_confidence_performance()
        
        return BacktestMetrics(
            total_bets=total_bets,
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            void_bets=void_bets,
            hit_rate=hit_rate,
            total_staked=total_staked,
            total_return=total_return,
            net_profit=net_profit,
            roi=roi,
            max_drawdown=self.max_drawdown * 100,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            turnover=total_staked / self.initial_bankroll,
            avg_bet_size=avg_bet_size,
            largest_win=largest_win,
            largest_loss=largest_loss,
            longest_winning_streak=streaks['longest_winning'],
            longest_losing_streak=streaks['longest_losing'],
            current_streak=streaks['current_length'],
            current_streak_type=streaks['current_type'],
            profit_by_month=profit_by_month,
            hit_rate_by_month=hit_rate_by_month,
            performance_by_market=performance_by_market,
            performance_by_confidence=performance_by_confidence
        )
    
    def _calculate_streaks(self) -> Dict[str, Any]:
        if not self.closed_positions:
            return {'longest_winning': 0, 'longest_losing': 0, 'current_length': 0, 'current_type': 'none'}
        
        sorted_positions = sorted(self.closed_positions, key=lambda x: x.timestamp)
        
        longest_winning = longest_losing = current_streak = 0
        current_type = 'none'
        temp_winning = temp_losing = 0
        
        for position in sorted_positions:
            if position.result == 'win':
                temp_winning += 1
                temp_losing = 0
                longest_winning = max(longest_winning, temp_winning)
            elif position.result == 'loss':
                temp_losing += 1
                temp_winning = 0
                longest_losing = max(longest_losing, temp_losing)
        
        if sorted_positions:
            last_result = sorted_positions[-1].result
            current_streak = temp_winning if last_result == 'win' else temp_losing
            current_type = last_result if last_result in ['win', 'loss'] else 'none'
        
        return {
            'longest_winning': longest_winning,
            'longest_losing': longest_losing,
            'current_length': current_streak,
            'current_type': current_type
        }
    
    def _calculate_monthly_performance(self) -> Dict[str, float]:
        monthly_profit = {}
        
        for position in self.closed_positions:
            if position.pnl is not None and position.close_timestamp:
                month_key = position.close_timestamp.strftime('%Y-%m')
                monthly_profit[month_key] = monthly_profit.get(month_key, 0) + position.pnl
        
        return monthly_profit
    
    def _calculate_monthly_hit_rates(self) -> Dict[str, float]:
        monthly_stats = {}
        
        for position in self.closed_positions:
            if position.close_timestamp and position.result in ['win', 'loss']:
                month_key = position.close_timestamp.strftime('%Y-%m')
                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {'wins': 0, 'total': 0}
                
                monthly_stats[month_key]['total'] += 1
                if position.result == 'win':
                    monthly_stats[month_key]['wins'] += 1
        
        return {
            month: (stats['wins'] / stats['total']) * 100
            for month, stats in monthly_stats.items()
            if stats['total'] > 0
        }
    
    def _calculate_market_performance(self) -> Dict[str, Dict[str, float]]:
        market_stats = {}
        
        for position in self.closed_positions:
            market = position.market_type
            if market not in market_stats:
                market_stats[market] = {'total_bets': 0, 'wins': 0, 'total_staked': 0, 'total_return': 0}
            
            stats = market_stats[market]
            stats['total_bets'] += 1
            stats['total_staked'] += position.stake
            
            if position.result == 'win':
                stats['wins'] += 1
            
            if position.pnl is not None:
                stats['total_return'] += position.pnl
        
        for market, stats in market_stats.items():
            stats['hit_rate'] = (stats['wins'] / stats['total_bets']) * 100 if stats['total_bets'] > 0 else 0
            stats['roi'] = (stats['total_return'] / stats['total_staked']) * 100 if stats['total_staked'] > 0 else 0
        
        return market_stats
    
    def _calculate_confidence_performance(self) -> Dict[str, Dict[str, float]]:
        confidence_buckets = {'50-60%': [], '60-70%': [], '70-80%': [], '80-90%': [], '90%+': []}
        
        for position in self.closed_positions:
            conf = position.prediction_confidence * 100
            
            if 50 <= conf < 60:
                bucket = '50-60%'
            elif 60 <= conf < 70:
                bucket = '60-70%'
            elif 70 <= conf < 80:
                bucket = '70-80%'
            elif 80 <= conf < 90:
                bucket = '80-90%'
            else:
                bucket = '90%+'
            
            confidence_buckets[bucket].append(position)
        
        bucket_stats = {}
        for bucket, positions in confidence_buckets.items():
            if not positions:
                continue
            
            total_bets = len(positions)
            wins = len([p for p in positions if p.result == 'win'])
            total_staked = sum(p.stake for p in positions)
            total_return = sum(p.pnl for p in positions if p.pnl is not None)
            
            bucket_stats[bucket] = {
                'total_bets': total_bets,
                'hit_rate': (wins / total_bets) * 100 if total_bets > 0 else 0,
                'roi': (total_return / total_staked) * 100 if total_staked > 0 else 0,
                'avg_stake': total_staked / total_bets if total_bets > 0 else 0
            }
        
        return bucket_stats
    
    async def _save_results(self, metrics: BacktestMetrics):
        run_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO backtest_runs 
            (run_id, start_date, end_date, initial_bankroll, final_bankroll, 
             total_bets, hit_rate, roi, sharpe_ratio, max_drawdown, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            self.closed_positions[0].timestamp.isoformat() if self.closed_positions else '',
            self.closed_positions[-1].close_timestamp.isoformat() if self.closed_positions else '',
            self.initial_bankroll,
            self.current_bankroll,
            metrics.total_bets,
            metrics.hit_rate,
            metrics.roi,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            datetime.now().isoformat()
        ))
        
        for position in self.closed_positions:
            conn.execute("""
                INSERT INTO backtest_positions 
                (position_id, run_id, match_id, team, market_type, stake, odds, 
                 timestamp, prediction_confidence, expected_value, result, pnl, close_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.position_id, run_id, position.match_id, position.team,
                position.market_type, position.stake, position.odds,
                position.timestamp.isoformat(), position.prediction_confidence,
                position.expected_value, position.result, position.pnl,
                position.close_timestamp.isoformat() if position.close_timestamp else None
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Backtest results saved with run_id: {run_id}")
