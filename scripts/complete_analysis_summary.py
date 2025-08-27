#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Analysis Summary - สรุปการวิเคราะห์ทุกระบบและคำแนะนำการเดิมพันที่ดีที่สุด
รวบรวมผลการวิเคราะห์จากทุกระบบและให้คำแนะนำการเดิมพันที่ดีที่สุด
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Windows console encoding fix
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class CompleteAnalysisSummary:
    def __init__(self):
        self.current_time = datetime.now().strftime("%H:%M:%S")
        
    def load_all_analysis_data(self) -> Dict[str, Any]:
        """โหลดข้อมูลการวิเคราะห์ทั้งหมด"""
        data_dir = "data"
        all_data = {}
        
        if not os.path.exists(data_dir):
            return all_data
            
        # โหลดข้อมูล Deep BLAST Analysis
        blast_files = [f for f in os.listdir(data_dir) if f.startswith("deep_blast_analysis_")]
        if blast_files:
            latest_blast = max(blast_files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
            try:
                with open(os.path.join(data_dir, latest_blast), 'r', encoding='utf-8') as f:
                    all_data['deep_blast'] = json.load(f)
            except:
                pass
                
        # โหลดข้อมูล VP vs GL Analysis
        vp_gl_files = [f for f in os.listdir(data_dir) if f.startswith("vp_gl_deep_analysis_")]
        if vp_gl_files:
            latest_vp_gl = max(vp_gl_files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
            try:
                with open(os.path.join(data_dir, latest_vp_gl), 'r', encoding='utf-8') as f:
                    all_data['vp_gl_deep'] = json.load(f)
            except:
                pass
                
        return all_data
        
    def display_complete_summary(self):
        """แสดงสรุปการวิเคราะห์ครบถ้วน"""
        
        print("="*100)
        print(f"🔬 สรุปการวิเคราะห์ครบถ้วนทุกระบบ - BLAST Open London 2025")
        print(f"⏰ เวลาปัจจุบัน: {self.current_time} | 🇹🇭 เวลาไทย")
        print("="*100)
        
        # โหลดข้อมูลทั้งหมด
        all_data = self.load_all_analysis_data()
        
        if not all_data:
            print("❌ ไม่พบข้อมูลการวิเคราะห์")
            return
            
        # แสดงการวิเคราะห์แต่ละแมตช์
        self.display_match_by_match_analysis(all_data)
        
        # แสดงสรุปการเดิมพันที่ดีที่สุด
        self.display_best_betting_recommendations(all_data)
        
        # แสดงกลยุทธ์การจัดการเงิน
        self.display_bankroll_strategy(all_data)
        
    def display_match_by_match_analysis(self, all_data: Dict):
        """แสดงการวิเคราะห์แต่ละแมตช์"""
        
        print("\n📊 การวิเคราะห์แต่ละแมตช์ (เรียงตามเวลา)")
        print("="*80)
        
        matches = [
            {"name": "Vitality vs M80", "time": "LIVE", "status": "🔴 กำลังแข่ง"},
            {"name": "Virtus.pro vs GamerLegion", "time": "20:30", "status": "🟡 กำลังจะเริ่ม"},
            {"name": "FaZe vs ECSTATIC", "time": "22:00", "status": "⏰ รอแข่ง"},
            {"name": "NAVI vs fnatic", "time": "00:30", "status": "⏰ รอแข่ง"}
        ]
        
        for i, match in enumerate(matches, 1):
            print(f"\n🎯 แมตช์ที่ {i}: {match['name']} ({match['time']}) {match['status']}")
            print("─" * 60)
            
            if match['name'] == "Virtus.pro vs GamerLegion" and 'vp_gl_deep' in all_data:
                self.display_vp_gl_detailed_analysis(all_data['vp_gl_deep'])
            elif 'deep_blast' in all_data:
                self.display_blast_match_analysis(match['name'], all_data['deep_blast'])
                
    def display_vp_gl_detailed_analysis(self, vp_gl_data: Dict):
        """แสดงการวิเคราะห์ VP vs GL แบบละเอียด"""
        
        print("🔍 การวิเคราะห์เชิงลึกพิเศษ:")
        
        # True Probabilities
        true_probs = vp_gl_data.get('true_probabilities', {})
        vp_prob = true_probs.get('vp', 0) * 100
        gl_prob = true_probs.get('gl', 0) * 100
        print(f"   🎯 ความน่าจะเป็นจริง: VP {vp_prob:.1f}% vs GL {gl_prob:.1f}%")
        
        # Map Analysis
        map_analysis = vp_gl_data.get('map_analysis', {})
        vp_maps = map_analysis.get('vp_strong_maps', [])
        gl_maps = map_analysis.get('gl_strong_maps', [])
        
        print(f"   🗺️ แมพที่เก่ง:")
        print(f"      VP: {', '.join([m['map'] for m in vp_maps])}")
        print(f"      GL: {', '.join([m['map'] for m in gl_maps])}")
        
        # Player Analysis
        player_analysis = vp_gl_data.get('player_analysis', {})
        carry_analysis = player_analysis.get('carry_analysis', {})
        if carry_analysis:
            vp_carry = carry_analysis.get('vp_top_carry', {})
            gl_carry = carry_analysis.get('gl_top_carry', {})
            print(f"   👤 ผู้เล่นหลัก: {vp_carry.get('player', 'N/A')} ({vp_carry.get('rating', 0)}) vs {gl_carry.get('player', 'N/A')} ({gl_carry.get('rating', 0)})")
        
        # Value Bets
        value_bets = vp_gl_data.get('value_bets', {})
        gl_bet = value_bets.get('gl')
        if gl_bet:
            kelly = gl_bet.get('kelly_fraction', 0) * 100
            print(f"   💰 Value Bet: GamerLegion ML @ 2.15 (Kelly: {kelly:.1f}%)")
            
    def display_blast_match_analysis(self, match_name: str, blast_data: Dict):
        """แสดงการวิเคราะห์จาก BLAST system"""
        
        analyses = blast_data.get('analyses', [])
        
        # หาแมตช์ที่ตรงกัน
        match_mapping = {
            "Vitality vs M80": "vitality_m80",
            "FaZe vs ECSTATIC": "faze_ecstatic", 
            "NAVI vs fnatic": "navi_fnatic"
        }
        
        match_id = match_mapping.get(match_name)
        if not match_id:
            print("   ❌ ไม่พบข้อมูลการวิเคราะห์")
            return
            
        match_data = None
        for analysis in analyses:
            if analysis.get('match_id') == match_id:
                match_data = analysis
                break
                
        if not match_data:
            print("   ❌ ไม่พบข้อมูลการวิเคราะห์")
            return
            
        # แสดงข้อมูลสำคัญ
        true_probs = match_data.get('true_probabilities', {})
        team1_prob = true_probs.get('team1', 0) * 100
        team2_prob = true_probs.get('team2', 0) * 100
        print(f"   🎯 ความน่าจะเป็นจริง: {team1_prob:.1f}% vs {team2_prob:.1f}%")
        
        # Value Bets
        value_bets = match_data.get('value_bets', [])
        if value_bets:
            for bet in value_bets:
                team = bet.get('team', 'Unknown')
                odds = bet.get('odds', 0)
                kelly = bet.get('kelly_fraction', 0) * 100
                print(f"   💰 Value Bet: {team} ML @ {odds} (Kelly: {kelly:.1f}%)")
        
        # Top Profit Opportunities
        opportunities = match_data.get('profit_opportunities', [])[:2]
        if opportunities:
            print("   🎲 โอกาสทำกำไรหลัก:")
            for opp in opportunities:
                category = opp.get('category', '')
                bet = opp.get('bet', '')
                probability = opp.get('probability', '')
                print(f"      • {category}: {bet} ({probability})")
                
    def display_best_betting_recommendations(self, all_data: Dict):
        """แสดงคำแนะนำการเดิมพันที่ดีที่สุด"""
        
        print(f"\n💎 คำแนะนำการเดิมพันที่ดีที่สุด")
        print("="*60)
        
        all_value_bets = []
        
        # รวบรวม Value Bets จากทุกระบบ
        if 'deep_blast' in all_data:
            analyses = all_data['deep_blast'].get('analyses', [])
            for analysis in analyses:
                value_bets = analysis.get('value_bets', [])
                for bet in value_bets:
                    bet['source'] = 'Deep BLAST'
                    bet['match_id'] = analysis.get('match_id', '')
                    all_value_bets.append(bet)
                    
        if 'vp_gl_deep' in all_data:
            vp_gl_bets = all_data['vp_gl_deep'].get('value_bets', {})
            if vp_gl_bets.get('gl'):
                bet = vp_gl_bets['gl'].copy()
                bet['team'] = 'GamerLegion'
                bet['odds'] = 2.15
                bet['source'] = 'VP-GL Deep'
                bet['match_id'] = 'vp_gl'
                all_value_bets.append(bet)
                
        # เรียงตาม Expected Value
        sorted_bets = sorted(all_value_bets, 
                           key=lambda x: x.get('expected_value', 0), 
                           reverse=True)
        
        print("🏆 Top Value Bets (เรียงตาม Expected Value):")
        for i, bet in enumerate(sorted_bets[:5], 1):
            team = bet.get('team', 'Unknown')
            odds = bet.get('odds', 0)
            kelly = bet.get('kelly_fraction', 0) * 100
            ev = bet.get('expected_value', 0)
            confidence = bet.get('confidence', 'Unknown')
            source = bet.get('source', 'Unknown')
            
            confidence_emoji = "🎯" if confidence == "High" else "📊" if confidence == "Medium" else "❓"
            print(f"   {i}. {confidence_emoji} {team} ML @ {odds}")
            print(f"      Kelly: {kelly:.1f}% | EV: {ev:.3f} | Source: {source}")
            
        # แนะนำการเดิมพันตามลำดับเวลา
        print(f"\n⏰ แนะนำการเดิมพันตามลำดับเวลา:")
        
        time_recommendations = [
            {"time": "ตอนนี้", "match": "VP vs GL", "bet": "GamerLegion ML @ 2.15", "kelly": "1.7%", "reason": "คู่ใกล้เคียง อัตราต่อรองดี"},
            {"time": "22:00", "match": "FaZe vs ECSTATIC", "bet": "FaZe ML @ 1.33", "kelly": "59.7%", "reason": "Value สูงสุด Edge ดีเยียม"},
            {"time": "00:30", "match": "NAVI vs fnatic", "bet": "NAVI ML @ 1.27", "kelly": "53.0%", "reason": "ปลอดภัยสุด เต็งแรง"}
        ]
        
        for rec in time_recommendations:
            print(f"   🕐 {rec['time']}: {rec['match']}")
            print(f"      💰 {rec['bet']} (Kelly: {rec['kelly']})")
            print(f"      📝 เหตุผล: {rec['reason']}")
            
    def display_bankroll_strategy(self, all_data: Dict):
        """แสดงกลยุทธ์การจัดการเงิน"""
        
        print(f"\n🎲 กลยุทธ์การจัดการเงิน (Bankroll Management)")
        print("="*60)
        
        print("💰 การแบ่งเงินทุน (สำหรับ Bankroll 10,000 บาท):")
        
        allocations = [
            {"match": "VP vs GL", "amount": "170 บาท", "percentage": "1.7%", "reason": "Kelly เล็ก แต่อัตราต่อรองดี"},
            {"match": "FaZe vs ECSTATIC", "amount": "3,000 บาท", "percentage": "30%", "reason": "Kelly สูงสุด แต่จำกัดความเสี่ยง"},
            {"match": "NAVI vs fnatic", "amount": "2,500 บาท", "percentage": "25%", "reason": "ปลอดภัย Kelly สูง"},
            {"match": "Map/Player Bets", "amount": "1,000 บาท", "percentage": "10%", "reason": "โบนัสจากการวิเคราะห์เชิงลึก"},
            {"match": "สำรอง", "amount": "3,330 บาท", "percentage": "33.3%", "reason": "รอโอกาสดีกว่าหรือ Live Betting"}
        ]
        
        for allocation in allocations:
            print(f"   • {allocation['match']}: {allocation['amount']} ({allocation['percentage']})")
            print(f"     └─ {allocation['reason']}")
            
        print(f"\n⚠️ หลักการสำคัญ:")
        print("   🛡️ ไม่เดิมพันเกิน 30% ในแมตช์เดียว")
        print("   📊 ใช้ Kelly Criterion แต่จำกัดความเสี่ยง")
        print("   ⏰ รอดูผลแมตช์แรกก่อนปรับกลยุทธ์")
        print("   🔄 หากชนะต่อเนื่อง เพิ่มเดิมพัน หากแพ้ ลดลง")
        print("   🚫 หยุดเดิมพันหากขาดทุนเกิน 20%")
        
        print(f"\n🎯 สรุปคำแนะนำสุดท้าย:")
        print("   1️⃣ เริ่มจาก VP vs GL (ความเสี่ยงต่ำ)")
        print("   2️⃣ FaZe vs ECSTATIC (เดิมพันหลัก)")
        print("   3️⃣ NAVI vs fnatic (ปิดท้ายปลอดภัย)")
        print("   4️⃣ Map Betting ตาม Veto ที่เห็น")
        
        end_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n⏰ เวลาสิ้นสุดการวิเคราะห์: {end_time}")
        print("="*100)

if __name__ == "__main__":
    summary = CompleteAnalysisSummary()
    summary.display_complete_summary()
