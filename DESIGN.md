# Daegu Bus Editable Retro Analysis Deck Design

## Purpose

This design system is for an editable PowerPoint deck about Daegu city bus stop demand and route supply imbalance. The deck may use GPT image slide planning language, but the final deliverable must be an editable PPTX, not full-slide bitmap pages.

## Visual System

- Theme name: Editable Retro Transit Dashboard
- Overall mood: analytical, civic, presentation-friendly, with restrained 1990s dashboard references
- Primary background: light neutral gray `#C0C0C0` or off-white chart panels
- Primary text: black `#000000`
- Main accent: navy `#000080`
- Secondary accent: blue `#1084D0`
- Warning accent: yellow `#FFFF00`
- Positive accent: green `#00AA00`
- Risk accent: red `#FF0000`
- Body slide density: medium-high, but never crammed
- No glassmorphism, no soft shadows, no decorative gradients beyond simple navy title strips

## Typography

- Korean title font: Malgun Gothic or Aptos Display fallback
- Korean body font: Malgun Gothic or Aptos fallback
- Numeric font: Aptos Mono, Consolas, or Courier New for KPI values
- Slide title: 30-38 pt
- Section label or kicker: 11-13 pt
- Body text: 16-20 pt
- Table text: 11-14 pt, only smaller when unavoidable
- KPI numbers: 30-48 pt
- Footnotes and source notes: 9-11 pt
- Avoid narrow multi-line title blocks. Keep slide titles short.

## Layout Principles

- Use a consistent header/body/footer system.
- Header zone: small section label, main slide title, optional one-line takeaway.
- Body zone: one dominant content system per slide, such as chart, table, matrix, pipeline, map, or KPI cards.
- Footer zone: source note, speaker section marker, or interpretation warning.
- Use explicit grid alignment. Do not scatter cards randomly.
- Prefer native editable PowerPoint objects:
  - titles and explanations as text boxes
  - KPI cards as shapes plus editable text
  - tables as native PowerPoint tables
  - charts as native PowerPoint charts
  - diagrams as native shapes and connectors
  - only maps and Streamlit screenshots as images

## Tightened Layout Mapping

Use these registered Tightened Slide layout ideas as planning references, while implementing the final deck as editable PPTX:

- S03 Split Statement: problem framing and synthesis
- S04 Six Cells: metric definition cards
- S07 Horizontal Bar: ranked stop or district bar charts
- S08 Duo Compare: paired charts or map plus explanation
- S11 Horizontal Timeline: data pipeline or workflow
- S15 Matrix + Hero Stat: correlation and candidate logic
- S16 Multi-card Brief: summary cards and grouped insights
- S17 System Diagram: analysis pipeline or policy roadmap
- S20 Stacked KPI Ledger: KPI-heavy dashboard pages
- S21 Tech Spec Sheet: data limitations and criteria tables
- S22 Image Hero: map or Streamlit screenshot only

## Chart Rules

- Charts must be editable PowerPoint charts unless using a captured map or app screenshot.
- Use clear Korean chart titles and axis labels.
- Use navy/blue as primary series, green/red only for comparison or increase/decrease.
- Add one clear annotation per chart when needed.
- Do not over-label every point.
- Do not imply causality from correlation.

## Table Rules

- Tables must be native editable PPT tables.
- Use Korean column labels, not raw CSV column names.
- Use thousands separators for counts.
- Use percent symbols for concentration and growth rates.
- Highlight only the most important row or column.
- Avoid raw dataframe appearance.

## Map and Screenshot Rules

- Maps and Streamlit screens may be inserted as images.
- Add editable callout labels above the image.
- Do not bake explanatory text into the screenshot.
- Keep map callouts limited to 2-4 representative examples.

## Icon and Diagram Rules

- Icons are optional and must clarify meaning.
- Use one consistent simple line icon style.
- Avoid random icon decoration.
- Diagrams use native shapes and straight or orthogonal connectors.

## Anti-patterns

- Do not create full-slide bitmap final pages.
- Do not paste tables or charts as screenshots unless there is no editable alternative.
- Do not invent data values.
- Do not state that weather or routes caused changes unless the evidence supports only correlation.
- Do not overcrowd slides with every candidate row.
- Do not use tiny unreadable table text.
