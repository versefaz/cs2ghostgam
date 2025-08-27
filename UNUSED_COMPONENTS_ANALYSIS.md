# CS2 Betting System - Unused Components Analysis

## 🔍 ส่วนประกอบที่ไม่ได้ใช้งาน (Unused Components)

### 📁 **Directories ที่ไม่ได้ใช้งาน**

#### 1. `cs2_betting_system/` - ระบบเก่าที่ซ้ำซ้อน
- **Status**: ไม่ได้ใช้งาน - มีระบบใหม่ใน `core/` แล้ว
- **Files**: 
  - `cs2_betting_system/main.py` (ซ้ำกับ `main.py`)
  - `cs2_betting_system/dashboard.py` (ไม่ได้ใช้)
  - `cs2_betting_system/integration.py` (ซ้ำกับ `core/integrated_pipeline.py`)
  - `cs2_betting_system/models/` (ซ้ำกับ `models/`)
  - `cs2_betting_system/scrapers/` (ซ้ำกับ `app/scrapers/`)

#### 2. `ml_pipeline/` - ML Pipeline เก่า
- **Status**: ไม่ได้ใช้งาน - ใช้ `core/ml/` แทน
- **Files**:
  - `ml_pipeline/model_trainer.py` (ไม่ได้ใช้)
  - `ml_pipeline/training/` (ไม่ได้ใช้)
  - `ml_pipeline/evaluation/` (ไม่ได้ใช้)

#### 3. `services/` - Microservices ที่ไม่ได้ Deploy
- **Status**: ไม่ได้ใช้งาน - ยังไม่ได้ใช้ microservices architecture
- **Directories**:
  - `services/api-gateway/`
  - `services/feature-builder/`
  - `services/feature_extractor/`
  - `services/feedback-logger/`
  - `services/hltv_scraper/`
  - `services/live-tracker/`
  - `services/model-trainer/`
  - `services/odds-scraper/`
  - `services/signal-generator/`

#### 4. `apps/desktop/` - Desktop App ที่ไม่สมบูรณ์
- **Status**: ไม่ได้ใช้งาน - ยังไม่เสร็จ
- **Files**: React/TypeScript components ที่ไม่ได้ build

#### 5. `infra/` - Infrastructure ที่ไม่ได้ Deploy
- **Status**: ไม่ได้ใช้งาน - ยังไม่ได้ deploy
- **Files**:
  - `infra/docker/`
  - `infra/helm/`
  - `infra/k8s/`

### 📄 **Files ที่ไม่ได้ใช้งาน**

#### Core Files
- `core/performance_optimizer.py` - ไม่ได้ใช้จริง
- `core/production_signal_generator.py` - ซ้ำกับ `signal_generator.py`
- `core/real_odds_scraper.py` - ไม่ได้ใช้
- `core/real_odds_api.py` - ไม่ได้ใช้
- `core/live_odds_fetcher.py` - ไม่ได้ใช้

#### App Files  
- `app/database.py` - ไม่ได้ใช้
- `app/models.py` - ไม่ได้ใช้
- `app/scrapers/vlr_scraper.py` - ไม่ได้ใช้ (Valorant)
- `app/scrapers/session_manager.py` - ซ้ำกับ `core/session/`

#### Scripts ที่ไม่จำเป็น
- `scripts/prepare_models.py` - ไม่ได้ใช้
- `scripts/monitor_hltv.py` - ไม่ได้ใช้
- `scripts/e2e_smoke_test.py` - ไม่ได้ใช้
- `scripts/match_predictor_cli.py` - ไม่ได้ใช้

#### Other Files
- `odds_manager.py` - ไม่ได้ใช้
- `example_usage.py` - ไม่ได้ใช้
- `apps/live_betting_terminal.py` - ไม่ได้ใช้
- `apps/real_odds_terminal.py` - ไม่ได้ใช้

### 🔧 **ส่วนประกอบที่ใช้งานจริง (Active Components)**

#### ✅ Core System
- `main.py` - Entry point หลัก
- `core/integrated_pipeline.py` - ระบบหลัก
- `core/session/session_manager.py` - จัดการ HTTP sessions
- `core/ml/feature_engineer.py` - Feature engineering
- `core/enhanced_team_analyzer.py` - วิเคราะห์ทีม
- `core/deep_betting_analyzer.py` - วิเคราะห์การเดิมพัน

#### ✅ Scrapers
- `app/scrapers/enhanced_hltv_scraper.py` - ดึงข้อมูล HLTV
- `app/scrapers/hltv_scraper.py` - HLTV scraper พื้นฐาน
- `app/scrapers/robust_odds_scraper.py` - ดึงราคาต่อรอง

#### ✅ Analysis Scripts
- `scripts/blast_london_analysis.py` - วิเคราะห์ BLAST London
- `scripts/triple_match_analysis.py` - วิเคราะห์ 3 แมตช์
- `scripts/enhanced_predictions_thai.py` - การทำนายภาษาไทย
- `scripts/gamerlegion_vs_virtuspro_analysis.py` - วิเคราะห์คู่เฉพาะ

#### ✅ Utilities
- `check_hltv_matches.py` - ตรวจสอบแมตช์
- `test_system.py` - ทดสอบระบบ

## 💡 **คำแนะนำการจัดระเบียบ**

### 🗑️ **ควรลบ (Safe to Delete)**
```bash
# Directories
rm -rf cs2_betting_system/
rm -rf ml_pipeline/
rm -rf services/
rm -rf apps/desktop/
rm -rf infra/docker/selenium-chrome/
rm -rf infra/helm/
rm -rf infra/k8s/

# Files
rm core/performance_optimizer.py
rm core/production_signal_generator.py
rm core/real_odds_*.py
rm core/live_odds_fetcher.py
rm app/database.py
rm app/models.py
rm app/scrapers/vlr_scraper.py
rm app/scrapers/session_manager.py
rm odds_manager.py
rm example_usage.py
rm apps/live_betting_terminal.py
rm apps/real_odds_terminal.py
```

### 📦 **ควรเก็บไว้ (Keep for Future)**
- `monitoring/` - สำหรับ monitoring ในอนาคต
- `risk_management/` - สำหรับ risk management
- `examples/` - ตัวอย่างการใช้งาน
- `tests/` - Unit tests

### 🔄 **ควรรวม (Consolidate)**
- รวม `modules/` เข้ากับ `core/`
- รวม `publishers/` เข้ากับ `core/pubsub/`
- รวม `pipeline/` เข้ากับ `core/`

## 📊 **สถิติการใช้งาน**

- **Total Files**: ~300 files
- **Active Files**: ~50 files (17%)
- **Unused Files**: ~250 files (83%)
- **Disk Space Saved**: ~50-100 MB หากลบไฟล์ที่ไม่ใช้

## ⚠️ **ข้อควรระวัง**

1. **Backup ก่อนลบ**: สร้าง backup ก่อนลบไฟล์
2. **ตรวจสอบ Dependencies**: ตรวจสอบว่าไม่มีไฟล์อื่นเรียกใช้
3. **Git History**: ไฟล์ที่ลบยังคงอยู่ใน Git history
4. **Future Use**: บางไฟล์อาจใช้ในอนาคต (microservices, desktop app)

## 🎯 **ผลลัพธ์หลังจัดระเบียบ**

- โครงสร้างโปรเจคชัดเจนขึ้น
- ลดความสับสน
- เพิ่มความเร็วในการ build/deploy
- ง่ายต่อการ maintenance
- ลด disk space usage
