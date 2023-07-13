"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=["companies", "shuttles", "reviews"],
                name="get_data",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="companies_prep",
                name="prep_companies",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="shuttles_prep",
                name="prep_shuttles",
            ),
            node(
                func=create_model_input_table,
                inputs=["shuttles_prep", "companies_prep", "reviews"],
                outputs="model_input_table",
                name="create_model_input_table",
            ),
        ]
    )
