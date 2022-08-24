from pydantic import BaseModel, Field


class Road(BaseModel):
    LinkID: str = Field(
        description="The ID of the road link according to DataMall.",
        example="103000000",
    )
    RoadCategory: str = Field(
        description="The category of the road according to DataMall. See https://datamall.lta.gov.sg/content/dam/datamall/datasets/LTA_DataMall_API_User_Guide.pdf for road category details.",
        example="E",
    )
    RoadName: str = Field(description="The name of the road")
    current_speedband: int = Field(
        description="The speedband of the road for the current_timestamp. See https://datamall.lta.gov.sg/content/dam/datamall/datasets/LTA_DataMall_API_User_Guide.pdf for mappings from speedband to speed.",
        example="KENT ROAD",
    )
    end_pos: str = Field(
        description="The latitude-longitude coordinates of the end position of the road.",
        example="1.3166840028663076 103.85259882242372",
    )
    length: float = Field(
        description="The length of the road in kilometers.", example=0.05611450062529966
    )
    predicted_speedbands: list[float] = Field(
        description="The predicted speedbands for the next 30 minutes. The first item is the 5 minute prediction, with each item being 5 minutes later up till the last item which is the 30 minute prediction.",
        example=[
            5.1742401123046875,
            4.770018577575684,
            4.586041450500488,
            4.5841145515441895,
            4.532934188842773,
            4.516937255859375,
        ],
    )
    start_pos: str = Field(
        description="The latitude-longitude coordinates of the start position of the road.",
        example="1.3170142376560023 103.85298052044503",
    )


class PredictionsResponse(BaseModel):
    current_timestamp: str = Field(
        description="The latest timestamp that the model has seen. Predictions will be for 5 to 30 mins from this timestamp.",
        example="18_07_2022_22_39_42",
    )
    roads: list[Road] = Field(
        description="The metadata and predictions for all roads supported by this prediction model."
    )
