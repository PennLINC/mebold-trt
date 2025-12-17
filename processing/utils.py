import numpy as np
import pandas as pd


def events_to_rtdur(events_df):
    """Implement Jeanette Mumford's ConsDurRTDur model on an events dataframe."""
    # Limit to 0back and 2back trials
    events_df = events_df.loc[events_df["trial_type"].isin(["0back", "2back"])]
    events_df = events_df[["onset", "duration", "trial_type", "response_time"]]

    # Implement Jeanette Mumford's ConsDurRTDur model
    # and rename trial types to valid Python identifiers:
    #   0back -> zero_back
    #   2back -> two_back
    events_df["trial_type"] = events_df["trial_type"].replace(
        {
            "0back": "zero_back",
            "2back": "two_back",
        }
    )
    # Long RTs are a lie!
    events_df.loc[events_df["response_time"] > 2, "response_time"] = np.nan

    response_events_df = events_df.loc[~np.isnan(events_df["response_time"])].copy()
    response_events_df.loc[:, "duration"] = response_events_df.loc[:, "response_time"]
    response_events_df.loc[:, "trial_type"] = "RTDur"
    cons_dur_rt_dur_events_df = pd.concat((events_df, response_events_df))
    cons_dur_rt_dur_events_df = cons_dur_rt_dur_events_df.sort_values(by="onset")
    return cons_dur_rt_dur_events_df
