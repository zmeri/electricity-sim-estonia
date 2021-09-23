#include <vector>

#include "storage_base.h"

using std::vector;


vector<double> storage_cpp(vector<double> demanded, vector<double> produced, double storage_power,
    double storage_energy, double eff_charge, double eff_discharge, double time_step) {
    int N = demanded.size();
    vector<double> stored(N, 0); // unit: MWs
    vector<double> to_storage(N, 0); // MW
    storage_energy = storage_energy * 3600; // MWs

    double stored_prev = 0;
    for (int i = 0; i < N; i++) {
        double s = produced[i] - demanded[i];
        if (stored_prev * eff_discharge + time_step * s < 0){
            if (stored_prev / time_step < storage_power){
                to_storage[i] = (-stored_prev * eff_discharge) / time_step;
            } else {
                to_storage[i] = -storage_power * eff_discharge;
            }
        } else if (stored_prev + s * eff_charge * time_step < storage_energy){
            if (s > 0){
                if (s < storage_power){
                    to_storage[i] = s;
                } else {
                    to_storage[i] = storage_power;
                }
            } else {
                if (abs(s) < storage_power){
                    to_storage[i] = s;
                } else {
                    to_storage[i] = -storage_power;
                }
            }
        } else if (stored_prev < storage_energy) {
            if ((storage_energy - stored_prev) / eff_charge / time_step < storage_power) {
                to_storage[i] = (storage_energy - stored_prev) / eff_charge / time_step;
            } else {
                to_storage[i] = storage_power;
            }
        }

        if (s > 0) {
            stored[i] = stored_prev + time_step * (eff_charge * to_storage[i]);
        } else {
            stored[i] = stored_prev + time_step * (to_storage[i] / eff_discharge);
        }

        stored_prev = stored[i];
    }

    return to_storage;
}
