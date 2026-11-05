-- Slotor schema v2: Track Architect export label per piece type (D18: schema changes are new migrations).
-- ta_label = the text of an export line before ", radius ..." (e.g. 'Curve 1/60 degrees'). Used to import inventories.
-- Only labels actually observed on an export are recorded; NULL = not observed yet.

ALTER TABLE piece_type ADD COLUMN ta_label TEXT;
