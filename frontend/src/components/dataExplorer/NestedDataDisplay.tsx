/**
 * Components for displaying nested/expanded data in DataExplorer tables.
 */
import type { NestedFieldConfig } from "./types";

// =============================================================================
// Types
// =============================================================================

interface NestedDataProps {
  row: Record<string, unknown>;
  nestedFields: NestedFieldConfig[];
}

interface NestedItemProps {
  item: Record<string, unknown>;
  fieldKey: string;
}

// =============================================================================
// Nested Data Container
// =============================================================================

export function NestedData({ row, nestedFields }: NestedDataProps) {
  return (
    <div className="nested-data">
      {nestedFields.map((field) => {
        const items = row[field.key] as Record<string, unknown>[] | undefined;
        if (!items || items.length === 0) return null;

        return (
          <div key={field.key} className="nested-data__section">
            <div className="nested-data__header">
              <span className="nested-data__icon">{field.icon}</span>
              <span className="nested-data__label">{field.label}</span>
              <span className="nested-data__count">({items.length})</span>
            </div>
            <div className="nested-data__items">
              {items.map((item, idx) => (
                <NestedItem key={idx} item={item} fieldKey={field.key} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// =============================================================================
// Nested Item Renderers
// =============================================================================

function NestedItem({ item, fieldKey }: NestedItemProps) {
  switch (fieldKey) {
    case "_private_texts":
      return <PrivateTextItem item={item} />;
    case "_descriptions":
      return <DescriptionItem item={item} />;
    case "_attachments":
      return <AttachmentItem item={item} />;
    case "_members":
      return <MemberItem item={item} />;
    default:
      return <GenericItem item={item} />;
  }
}

function PrivateTextItem({ item }: { item: Record<string, unknown> }) {
  return (
    <div className="nested-item nested-item--text">
      <div className="nested-item__type">
        {String(item.metadata_type || item.type || "note")}
      </div>
      <div className="nested-item__content">{String(item.text || "")}</div>
      <div className="nested-item__meta">
        {item.metadata_file_name ? (
          <span>📎 {String(item.metadata_file_name)}</span>
        ) : null}
        {item.metadata_created_at ? (
          <span>🕐 {String(item.metadata_created_at)}</span>
        ) : null}
      </div>
    </div>
  );
}

function DescriptionItem({ item }: { item: Record<string, unknown> }) {
  return (
    <div className="nested-item nested-item--description">
      <div className="nested-item__title">
        {String(item.title || "Opportunity Notes")}
      </div>
      <div className="nested-item__content">{String(item.text || "")}</div>
      <div className="nested-item__meta">
        {item.created_at ? <span>Created: {String(item.created_at)}</span> : null}
      </div>
    </div>
  );
}

function AttachmentItem({ item }: { item: Record<string, unknown> }) {
  return (
    <div className="nested-item nested-item--attachment">
      <span className="nested-item__icon">📎</span>
      <span className="nested-item__filename">
        {String(item.file_name || item.name || "Attachment")}
      </span>
      <span className="nested-item__size">{String(item.file_size || "")}</span>
    </div>
  );
}

function MemberItem({ item }: { item: Record<string, unknown> }) {
  return (
    <div className="nested-item nested-item--member">
      <span className="nested-item__icon">🏢</span>
      <span className="nested-item__id">
        Company: {String(item.company_id || "")}
      </span>
      {item.contact_id ? (
        <span className="nested-item__contact">
          Contact: {String(item.contact_id)}
        </span>
      ) : null}
      <span className="nested-item__date">
        Added: {String(item.added_at || "")}
      </span>
    </div>
  );
}

function GenericItem({ item }: { item: Record<string, unknown> }) {
  return (
    <div className="nested-item">
      {Object.entries(item)
        .filter(([key]) => !key.startsWith("_"))
        .slice(0, 5)
        .map(([key, val]) => (
          <span key={key} className="nested-item__field">
            <strong>{key}:</strong> {String(val)}
          </span>
        ))}
    </div>
  );
}
