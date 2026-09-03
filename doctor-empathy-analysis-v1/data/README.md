# Data File Guide

## Dataset Overview

This directory contains all raw inputs and sample data required for the project analyses.

## File Inventory

### Primary Data Assets
- **Sample Data.xlsx** – Medical consultation samples, including doctor–patient dialogue transcripts
- **detailed_empathy_analysis.json** – Detailed empathy analysis results in JSON format

## 🔍 Data Format Details

### Excel Structure
- **File**: `Sample Data.xlsx`
- **Content**: Consultation transcripts
- **Columns**:  
  - Column 5 (index 4): Dialogue text  
  - Remaining columns: Metadata fields
- **Purpose**: Core analysis source

### JSON Analysis Output
- **File**: `detailed_empathy_analysis.json`
- **Content**: Empathy feature analysis results
- **Format**: Structured JSON
- **Purpose**: Detailed record of analytical outputs

## 📋 Data Requirements

### Input Specifications
- **File types**: Excel (.xlsx) or CSV (.csv)
- **Encoding**: UTF-8
- **Language**: Chinese medical dialogue text
- **Content**: Doctor and patient utterances

### Data Quality Guidelines
- Complete text content
- Clear conversational logic
- Presence of empathetic expressions
- Consistent formatting

## 🚀 Usage

### Run the Analysis
```bash
# From the project root
python src/empathy_analysis.py

# From inside src
cd src
python empathy_analysis.py
```

### Use Custom Data
To analyze your own dataset:
1. Place the file inside the `data/` directory
2. Ensure the format matches the requirements
3. Update file paths in code if necessary
4. Run the analysis script

## 📝 Data Preprocessing

### Automated Steps
The pipeline automatically performs:
- Text cleaning
- Doctor utterance extraction
- Empathy feature detection
- Score computation

### Manual Preparation
If manual preprocessing is required:
1. Validate data completeness
2. Remove invalid records
3. Standardize formatting
4. Verify quality

## 🔒 Data Security

### Privacy Protection
- Medical data is sensitive
- Use de-identified datasets
- Avoid personal identifiers
- Comply with applicable regulations

### Backups
- Regularly back up raw data
- Preserve analysis artifacts
- Maintain version control
- Prevent data loss

## 📊 Data Statistics

### Sample Size
- Current volume: Determined by provided files
- Recommended minimum: 100+ dialogue cases
- Absolute minimum: 10+ dialogue cases

### Distribution Tracking
- Dialogue length distribution
- Empathy feature distribution
- Doctor communication patterns

---

*Last updated: 2024*
