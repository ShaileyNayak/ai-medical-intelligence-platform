from sqlalchemy.orm import Session

from app.models.db_models import Prediction


def create_prediction(
    db: Session,
    *,
    image_filename: str,
    predicted_label: str,
    confidence_score: float,
    heatmap_path: str,
    llm_report: str,
    model_version: str,
    user_id: int | None = None,
) -> Prediction:
    record = Prediction(
        image_filename=image_filename,
        predicted_label=predicted_label,
        confidence_score=confidence_score,
        heatmap_path=heatmap_path,
        llm_report=llm_report,
        model_version=model_version,
        user_id=user_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_predictions(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Prediction], int]:
    query = db.query(Prediction)
    total = query.count()
    items = (
        query.order_by(Prediction.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def get_prediction(db: Session, prediction_id: int) -> Prediction | None:
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


def delete_prediction(db: Session, prediction_id: int) -> bool:
    record = get_prediction(db, prediction_id)
    if not record:
        return False
    db.delete(record)
    db.commit()
    return True


def update_report(db: Session, prediction_id: int, llm_report: str) -> Prediction | None:
    record = get_prediction(db, prediction_id)
    if not record:
        return None
    record.llm_report = llm_report
    db.commit()
    db.refresh(record)
    return record
