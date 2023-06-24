import mne


def _find_events(raw, event_id):
    # find the events, first check stim_channels then annotations
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    if len(stim_channels) > 0:
        events = mne.find_events(raw, shortest_event=0, verbose=False)
    else:
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    return events
