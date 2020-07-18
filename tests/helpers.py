from cargonet.constants.numbers import (
    MININT,
    STATUS_ARRIVAL,
    STATUS_DEPARTURE,
    STATUS_DRIVE_THROUGH,
)

END_STATION_ID = 9999


def flatten(l):
    return [item for sublist in l for item in sublist]


def delay(delta):
    return int(delta.total_seconds() / 60.0)


def station(
    station_id,
    planned_arrival=None,
    planned_departure=None,
    planned_event_time=None,
    arrival=None,
    departure=None,
    event_time=None,
):
    planned, live = [], []

    if planned_event_time:
        assert not planned_arrival and not planned_departure
    if event_time:
        assert not arrival and not departure

    if planned_arrival:
        planned.append(
            {
                "endStationId": END_STATION_ID,
                "plannedEventTime": planned_arrival,
                "stationId": station_id,
                "status": STATUS_ARRIVAL,
            }
        )
    if planned_departure:
        planned.append(
            {
                "endStationId": END_STATION_ID,
                "plannedEventTime": planned_departure,
                "stationId": station_id,
                "status": STATUS_DEPARTURE,
            }
        )
    if event_time:
        planned.append(
            {
                "endStationId": END_STATION_ID,
                "plannedEventTime": event_time,
                "stationId": station_id,
                "status": STATUS_DRIVE_THROUGH,
            }
        )

    if arrival:
        live.append(
            {
                "delay": delay(
                    arrival
                    - (planned_arrival or planned_event_time or planned_departure)
                ),
                "eventTime": arrival,
                "stationId": station_id,
                "status": STATUS_ARRIVAL,
            }
        )
    if departure:
        live.append(
            {
                "delay": delay(
                    departure
                    - (planned_departure or planned_event_time or planned_arrival)
                ),
                "eventTime": departure,
                "stationId": station_id,
                "status": STATUS_DEPARTURE,
            }
        )
    if event_time:
        live.append(
            {
                "delay": delay(
                    event_time
                    - (planned_event_time or planned_departure or planned_event_time)
                ),
                "eventTime": event_time,
                "stationId": station_id,
                "status": STATUS_DRIVE_THROUGH,
            }
        )
    return planned, live


def create_route(transport_id, *entries):
    planned, live = flatten([p for p, l in entries]), flatten([l for p, l in entries])
    return dict(
        live=live,
        planned=planned,
        transport_id=transport_id,
        endStationId=END_STATION_ID,
        plannedArrivalTimeEndStation=planned[-1].get("plannedEventTime"),
    )
