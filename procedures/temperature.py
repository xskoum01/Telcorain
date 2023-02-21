import numpy as np
import pandas as pd
import xarray as xr


class Temperature:

    def unit_temperature_a(self, link):
        temperature_a = np.array(link['temperature_tx'])
        prsi = False
        if temperature_a > 40:
            prsi = False

        else:
            prsi = True

    def temperature(self, link):
        calc_data = []
        link_channels = []
        curr_link = 0
        count = 0
        link_count = 0
        temperature_tx = np.array(link['temperature_tx'])

        calc_data.append(xr.concat(link_channels, dim="channel_id"))

        # for link in calc_data:
        #     link['temperature_rx'] = link.temperature_rx.astype(float).interpolate_na(dim='time', method='linear',
        #                                                                               max_gap='5min')
            # link['temperature_rx'] = link.temperature_rx.astype(float).fillna(0.0)

            # link['temperature_tx'] = link.temperature_tx.astype(float).interpolate_na(dim='time', method='linear',
            #                                                                           max_gap='5min')
            # link['temperature_tx'] = link.temperature_tx.astype(float).fillna(0.0)

            # self.sig.progress_signal.emit({'prg_val': round((curr_link / link_count) * 15) + 35})

            # print("Ted neco uvidis")
            # print("Teplota tx")
            # print(link['temperature_tx'])

            # curr_link += 1
            # count += 1

