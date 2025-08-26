#!/usr/bin/env python3
"""
Discord Bot Consumer for CS2 Betting Signals
Subscribes to Redis channels and posts betting signals to Discord
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

import aioredis
import discord
from discord.ext import commands

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CS2BettingBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.redis = None
        self.channels = {
            'signals': os.getenv('REDIS_SIGNALS_CHANNEL', 'cs2:signals'),
            'predictions': os.getenv('REDIS_PREDICTIONS_CHANNEL', 'cs2:predictions'),
            'alerts': os.getenv('REDIS_ALERTS_CHANNEL', 'cs2:alerts')
        }
        
    async def setup_hook(self):
        """Initialize Redis connection and start consumers"""
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis = aioredis.from_url(redis_url)
        
        # Start Redis consumers
        for channel_name, channel_key in self.channels.items():
            self.loop.create_task(self.consume_channel(channel_name, channel_key))
        
        logger.info("CS2 Betting Bot initialized and consumers started")
    
    async def consume_channel(self, channel_name: str, channel_key: str):
        """Consumer for specific Redis channel"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel_key)
            
            logger.info(f"Subscribed to {channel_key}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self.handle_message(channel_name, data)
                    except Exception as e:
                        logger.error(f"Error processing message from {channel_key}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in consumer for {channel_key}: {e}")
            # Retry after delay
            await asyncio.sleep(5)
            self.loop.create_task(self.consume_channel(channel_name, channel_key))
    
    async def handle_message(self, channel_name: str, data: Dict[str, Any]):
        """Handle incoming Redis messages"""
        if channel_name == 'signals':
            await self.handle_signal(data)
        elif channel_name == 'predictions':
            await self.handle_prediction(data)
        elif channel_name == 'alerts':
            await self.handle_alert(data)
    
    async def handle_signal(self, signal: Dict[str, Any]):
        """Handle betting signals"""
        embed = discord.Embed(
            title="🎯 New Betting Signal",
            color=0x00ff00 if signal.get('action') == 'BET' else 0xff0000,
            timestamp=datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
        )
        
        # Match info
        match = signal.get('match', {})
        embed.add_field(
            name="Match",
            value=f"{match.get('team1', 'Team1')} vs {match.get('team2', 'Team2')}",
            inline=False
        )
        
        # Signal details
        embed.add_field(name="Action", value=signal.get('action', 'UNKNOWN'), inline=True)
        embed.add_field(name="Market", value=signal.get('market', 'match_winner'), inline=True)
        embed.add_field(name="Selection", value=signal.get('selection', 'N/A'), inline=True)
        
        # Financial info
        embed.add_field(name="Odds", value=f"{signal.get('odds', 0):.2f}", inline=True)
        embed.add_field(name="Stake", value=f"${signal.get('stake', 0):.2f}", inline=True)
        embed.add_field(name="Expected Value", value=f"{signal.get('expected_value', 0):.3f}", inline=True)
        
        # Confidence and edge
        embed.add_field(name="Confidence", value=f"{signal.get('confidence', 0):.1%}", inline=True)
        embed.add_field(name="Edge", value=f"{signal.get('edge', 0):.2%}", inline=True)
        embed.add_field(name="Kelly %", value=f"{signal.get('kelly_fraction', 0):.2%}", inline=True)
        
        # Send to configured channel
        channel_id = int(os.getenv('DISCORD_SIGNALS_CHANNEL_ID', '0'))
        if channel_id:
            channel = self.get_channel(channel_id)
            if channel:
                await channel.send(embed=embed)
    
    async def handle_prediction(self, prediction: Dict[str, Any]):
        """Handle match predictions"""
        embed = discord.Embed(
            title="🔮 Match Prediction",
            color=0x0099ff,
            timestamp=datetime.fromisoformat(prediction.get('timestamp', datetime.now().isoformat()))
        )
        
        # Match info
        match = prediction.get('match', {})
        embed.add_field(
            name="Match",
            value=f"{match.get('team1', 'Team1')} vs {match.get('team2', 'Team2')}",
            inline=False
        )
        
        # Prediction details
        embed.add_field(name="Predicted Winner", value=prediction.get('predicted_winner', 'N/A'), inline=True)
        embed.add_field(name="Confidence", value=f"{prediction.get('confidence', 0):.1%}", inline=True)
        embed.add_field(name="Model", value=prediction.get('model_version', 'v1.0'), inline=True)
        
        # Probabilities
        probs = prediction.get('probabilities', {})
        if probs:
            prob_text = "\n".join([f"{team}: {prob:.1%}" for team, prob in probs.items()])
            embed.add_field(name="Win Probabilities", value=prob_text, inline=False)
        
        # Send to predictions channel
        channel_id = int(os.getenv('DISCORD_PREDICTIONS_CHANNEL_ID', '0'))
        if channel_id:
            channel = self.get_channel(channel_id)
            if channel:
                await channel.send(embed=embed)
    
    async def handle_alert(self, alert: Dict[str, Any]):
        """Handle system alerts"""
        severity = alert.get('severity', 'info').upper()
        colors = {
            'CRITICAL': 0xff0000,
            'ERROR': 0xff6600,
            'WARNING': 0xffff00,
            'INFO': 0x00ff00
        }
        
        embed = discord.Embed(
            title=f"🚨 {severity} Alert",
            description=alert.get('message', 'System alert'),
            color=colors.get(severity, 0x808080),
            timestamp=datetime.fromisoformat(alert.get('timestamp', datetime.now().isoformat()))
        )
        
        embed.add_field(name="Source", value=alert.get('source', 'System'), inline=True)
        embed.add_field(name="Type", value=alert.get('alert_type', 'general'), inline=True)
        
        if alert.get('details'):
            embed.add_field(name="Details", value=str(alert['details'])[:1024], inline=False)
        
        # Send to alerts channel
        channel_id = int(os.getenv('DISCORD_ALERTS_CHANNEL_ID', '0'))
        if channel_id:
            channel = self.get_channel(channel_id)
            if channel:
                await channel.send(embed=embed)
    
    @commands.command(name='status')
    async def status_command(self, ctx):
        """Get system status"""
        try:
            # Get Redis stats
            redis_info = await self.redis.info()
            
            embed = discord.Embed(
                title="📊 CS2 Betting System Status",
                color=0x00ff00,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Bot Status", value="🟢 Online", inline=True)
            embed.add_field(name="Redis", value="🟢 Connected", inline=True)
            embed.add_field(name="Uptime", value=f"{redis_info.get('uptime_in_seconds', 0)}s", inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error getting status: {e}")
    
    @commands.command(name='recent')
    async def recent_signals(self, ctx, limit: int = 5):
        """Get recent signals"""
        try:
            # Get recent signals from Redis
            signals_key = "cs2:signals:recent"
            signals = await self.redis.lrange(signals_key, 0, limit - 1)
            
            if not signals:
                await ctx.send("No recent signals found.")
                return
            
            embed = discord.Embed(
                title=f"📈 Recent {len(signals)} Signals",
                color=0x0099ff,
                timestamp=datetime.now()
            )
            
            for i, signal_data in enumerate(signals, 1):
                try:
                    signal = json.loads(signal_data)
                    match = signal.get('match', {})
                    embed.add_field(
                        name=f"Signal {i}",
                        value=f"{match.get('team1', 'Team1')} vs {match.get('team2', 'Team2')}\n"
                              f"Action: {signal.get('action', 'N/A')}\n"
                              f"Odds: {signal.get('odds', 0):.2f}",
                        inline=True
                    )
                except json.JSONDecodeError:
                    continue
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error getting recent signals: {e}")

async def main():
    """Main function to run the bot"""
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        logger.error("DISCORD_BOT_TOKEN environment variable not set")
        return
    
    bot = CS2BettingBot()
    
    try:
        await bot.start(bot_token)
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        if bot.redis:
            await bot.redis.close()

if __name__ == "__main__":
    asyncio.run(main())
