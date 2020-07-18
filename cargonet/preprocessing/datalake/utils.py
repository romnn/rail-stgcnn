def pack_transport_sections(transport):
    def find_latest(t):
        t = [
            t
            for t in t
            if None
            not in [
                t.get("plannedDepartureTime"),
                t.get("plannedArrivalTime"),
                t.get("plannedDepartureTimeStartStation"),
                t.get("plannedArrivalTimeEndStation"),
            ]
        ]
        _remaining, _latest = [], []
        sorted_production_dates = sorted([sec.get("prodDate") for sec in t])
        if len(sorted_production_dates) < 1:
            return _remaining, _latest

        latest_production_date = sorted_production_dates[-1]
        for r in t:
            if r.get("prodDate") == latest_production_date:
                _latest.append(r)
            else:
                _remaining.append(r)
        # make sure latest is sorted
        _latest = sorted(_latest, key=lambda sec: sec.get("plannedDepartureTime"))
        return _remaining, _latest

    def equal(a, b, *keys):
        return all([a.get(k) == b.get(k) for k in keys])

    # Packing
    packed = []
    remaining, latest = find_latest(transport)
    packed = latest
    while True:
        done = True
        for r in remaining:
            # Try to pack if has the same start and end station
            if all([equal(l, r, "startStationId", "endStationId") for l in packed]):
                continue
            # Check as new first element
            if packed[0].get("plannedDepartureTime") < packed[0].get(
                "plannedDepartureTimeStartStation"
            ):
                if r.get("plannedArrivalTime") < packed[0].get("plannedDepartureTime"):
                    packed = [r] + packed
                    done = False
                    continue
            # Check as new last element
            if packed[-1].get("plannedArrivalTime") < packed[-1].get(
                "plannedArrivalTimeEndStation"
            ):
                if r.get("plannedDepartureTime") > packed[-1].get("plannedArrivalTime"):
                    packed = packed + [r]
                    done = False
                    continue
            # Check inbetween
            for i in range(1, len(packed)):
                pl, pr = packed[i - 1], packed[i]
                if (
                    pl.get("plannedArrivalTime")
                    <= r.get("plannedDepartureTime")
                    <= r.get("plannedArrivalTime")
                    <= pr.get("plannedArrivalTime")
                ):
                    packed = packed[:i] + [r] + packed[i:]
                    done = False
                    break

        if done:
            break

        # Continue with the next iteration
        remaining, latest = find_latest(remaining)
    return packed
