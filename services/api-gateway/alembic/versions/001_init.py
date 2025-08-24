"""initial placeholder migration"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_init'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # TODO: apply real DDL here (TimescaleDB schema)
    pass


def downgrade() -> None:
    # TODO: drop objects created in upgrade
    pass
