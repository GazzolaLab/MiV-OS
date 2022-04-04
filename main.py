from spike_analysis import *

def run():
    t, raw = load_data(data_file='continuous.dat', channels=64)
    a = 30
    b = 90
    k_t = 8.5
    x = filter_data(raw, 600, 2500)
    #plot_channels(t, raw, 'd_m_div_lfp_'+str(a)+'_to_'+str(b)+'s', title='Day 3 in Media 3, Day 19 IV: LFP (-125\u03BCV to 125\u03BCV)', t_i=a, t_f=b)
    #plot_channels(t, x, 'd_m_div_filtered_'+str(a)+'_to_'+str(b)+'s', title='Day 3 in Media 3, Day 19 IV: Filtered Neural Recording (-300\u03BCV to 300\u03BCV)', t_i=a, t_f=b)
    spike_data = get_spikes(x, k_t, t_i=a, t_f=b)
    plot_channels(t, x, 'd15_control2_div_spike_detection_'+str(a)+'_to_'+str(b)+'s', title='Control 2, Day 15 IV: Spike Detection (-300\u03BCV to 300\u03BCV)', k=k_t, raster_on=True, spikes=spike_data, t_i=a, t_f=b)
    rasterize(spike_data,'d27_m4_div_raster_'+str(a)+'_to_'+str(b)+'s', title='Day 11 in Media 4, Day 27 IV: Raster Plot')
    max_rate, mean_rate, rate_variance = get_spike_stats(spike_data, t_i=a, t_f=b)

    with open('spike_stats.txt', 'a') as f:
        f.write('['+str(max_rate[0])+', '+str(mean_rate)+', '+str(rate_variance)+'], ')

    #crosscorrelate(x, t_i=a, t_f=b)
    #plotnetworkconnectivity

def plot():
    t = [15,19,21,23,25,27]

    controls = [[21.45, 1.141111111111111, 19.35525432098765], [13.3, 0.5094444444444445, 5.697401543209877],[0, 0, 0], [0, 0, 0]]
    day19 = [[0.5, 0.03722222222222223, 0.007845987654320987], [69.66666666666667, 6.257222222222223, 253.090975617284],[52.86666666666667, 9.134444444444444, 204.30229506172836],[0.6, 0.22888888888888895, 0.0061839506172839485]]
    day21 = [[0.75, 0.15722222222222224, 0.021197839506172836], [34.65, 4.373333333333333, 77.9726962962963],[42.78333333333333, 7.674999999999999, 108.87014351851852],[16.516666666666666, 0.9077777777777777, 8.898550617283949]]
    day23 = [[11.616666666666667, 0.38722222222222225, 4.348290432098764],
             [74.91666666666667, 6.00111111111111, 213.67733209876542],
             [78.86666666666666, 11.261111111111111, 309.74619135802465],
             [1.8666666666666667, 0.09222222222222222, 0.13366172839506174]]
    day25 = [[21.766666666666666, 0.7516666666666667, 15.241895370370369],
             [14.5, 1.5666666666666667, 11.365277777777777], [96.75, 17.139444444444443, 643.0030459876543],
             [0.6833333333333333, 0.02277777777777778, 0.015045987654320992]]
    day27 = [[10.883333333333333, 0.38, 3.808803703703703], [41.93333333333333, 3.741111111111111, 72.94097654320989],
             [74.15, 18.05666666666667, 507.55464074074075], [9.85, 1.3744444444444446, 7.498976543209879]]

    var_m1 = [controls[0][2], day19[0][2], day21[0][2], day23[0][2], day25[0][2], day27[0][2]]
    var_m2 = [controls[1][2], day19[1][2], day21[1][1], day23[1][2], day25[1][2], day27[1][2]]
    var_m3 = [controls[2][2], day19[2][2], day21[2][1], day23[2][2], day25[2][2], day27[2][2]]
    var_m4 = [controls[3][2], day19[3][2], day21[3][1], day23[3][2], day25[3][2], day27[3][2]]

    plt.xlabel('Time (days)')
    plt.title('Firing Rate Variance')
    plt.plot(t, var_m1)
    plt.plot(t, var_m2)
    plt.plot(t, var_m3)
    plt.plot(t, var_m4)
    plt.legend(['Media 1','Media 2','Media 3','Media 4'])
    plt.savefig('var_rate_over_days.png')
    plt.show()


if __name__ == '__main__':
    #run()
    plot()