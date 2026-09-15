"""Official field-level schema for the Ali-CCP public dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FieldSpec:
    field_id: str
    name: str
    domain: str
    description_zh: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


FIELD_SPECS = (
    FieldSpec("101", "user_id", "user", "用户 ID"),
    FieldSpec("109_14", "user_category_history", "user", "用户商品类目历史行为及累计数量"),
    FieldSpec("110_14", "user_shop_history", "user", "用户商品店铺历史行为及累计数量"),
    FieldSpec("127_14", "user_brand_history", "user", "用户商品品牌历史行为及累计数量"),
    FieldSpec("150_14", "user_intention_history", "user", "用户意图历史行为及累计数量"),
    FieldSpec("121", "user_type_1", "user", "用户分类 ID（类型一）"),
    FieldSpec("122", "user_type_2", "user", "用户分类 ID（类型二）"),
    FieldSpec("124", "user_gender", "user", "用户性别分类 ID"),
    FieldSpec("125", "user_age", "user", "用户年龄分类 ID"),
    FieldSpec("126", "user_consumption_1", "user", "用户消费水平分类（类型一）"),
    FieldSpec("127", "user_consumption_2", "user", "用户消费水平分类（类型二）"),
    FieldSpec("128", "user_employment", "user", "用户是否就业"),
    FieldSpec("129", "user_geography", "user", "用户地理信息分类 ID"),
    FieldSpec("205", "item_id", "item", "商品 ID"),
    FieldSpec("206", "item_category", "item", "商品所属类目 ID"),
    FieldSpec("207", "item_shop", "item", "商品所属店铺 ID"),
    FieldSpec("210", "item_intention", "item", "商品关联用户意图 ID"),
    FieldSpec("216", "item_brand", "item", "商品品牌 ID"),
    FieldSpec("508", "user_item_category_cross", "combination", "109_14 与 206 的组合特征"),
    FieldSpec("509", "user_item_shop_cross", "combination", "110_14 与 207 的组合特征"),
    FieldSpec("702", "user_item_brand_cross", "combination", "127_14 与 216 的组合特征"),
    FieldSpec("853", "user_item_intention_cross", "combination", "150_14 与 210 的组合特征"),
    FieldSpec("301", "context_scene", "context", "业务场景分类"),
)

FIELD_SPEC_BY_ID = {spec.field_id: spec for spec in FIELD_SPECS}
ALL_FIELD_IDS = tuple(spec.field_id for spec in FIELD_SPECS)
COMMON_FIELD_IDS = tuple(spec.field_id for spec in FIELD_SPECS if spec.domain == "user")
SAMPLE_FIELD_IDS = tuple(spec.field_id for spec in FIELD_SPECS if spec.domain != "user")
EXPECTED_COMMON_FIELDS = frozenset(COMMON_FIELD_IDS)
EXPECTED_SAMPLE_FIELDS = frozenset(SAMPLE_FIELD_IDS)
