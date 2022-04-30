---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

#

## References

- Spike sorting based on discrete wavelet transform coefficients (Letelier 2000)
- Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering (Quiroga 2004)
- A novel and fully automatic spike-sorting implementation with variable number of features (Chaure 2018)

```{code-cell} ipython3
:tags: [hide-cell]

import os, sys
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import pywt
```

```{code-cell} ipython3
:tags: [remove-cell]

sys.path.append('../..')
```

```{code-cell} ipython3
:tags: [remove-cell]

from miv.io import load_continuous_data_file

datapath = '2022-03-10_16-19-09/Record Node 104/spontaneous/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat'
rate = 30_000
timestamps, signal = load_continuous_data_file(datapath, 64, rate)
```

## Pre-Filter

```{code-cell} ipython3
:tags: [hide-cell]

from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
```

```{code-cell} ipython3
pre_filter = ButterBandpass(lowcut=300, highcut=3000, order=5)
filtered_signal = pre_filter(signal, sampling_rate=rate)

spike_detection = ThresholdCutoff()
spks = spike_detection(filtered_signal, timestamps, sampling_rate=30_000, progress_bar=False)
```

## Plot

```{code-cell} ipython3
:tags: [hide-cell]

from miv.visualization import extract_waveforms, plot_waveforms
```

```{code-cell} ipython3
cutouts = extract_waveforms(
    filtered_signal, spks, channel=7, sampling_rate=rate
)
plot_waveforms(cutouts, rate, n_spikes=250)
```

## Wavelet Decomposition

```{raw-cell}
tmp_list = []
for i in range(n_components):
    idx = labels == i
    tmp_list.append(timestamps[idx])
    spikestamps_clustered.append(tmp_list)

_ = plt.figure(figsize=(8, 8))
for i in range(n_components):
    idx = labels == i
    _ = plt.plot(transformed[idx, 0], transformed[idx, 1], ".")
    _ = plt.title("Cluster assignments by a GMM")
    _ = plt.xlabel("Principal Component 1")
    _ = plt.ylabel("Principal Component 2")
    _ = plt.legend([0, 1, 2])
    _ = plt.axis("tight")

_ = plt.figure(figsize=(8, 8))
for i in range(n_components):
    idx = labels == i
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
    plot_waveforms(
        cutouts[idx, :], rate, n_spikes=100, color=color,
    )
# custom legend
custom_lines = [plt.Line2D([0], [0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i], lw=4,) \
                    for i in range(n_components)]
plt.legend(custom_lines, [f"component {i}" for i in range(n_components)])
```

```{code-cell} ipython3
spikes_l = cutouts[0]
coeffs = pywt.wavedec(cutouts, 'haar', level=4)
features = np.concatenate(coeffs, axis=1)b
```

```{code-cell} ipython3
# Kalmogorov S Test
function [KSmax] = test_ks(x)
%
% Calculates the CDF (expcdf)
%[y_expcdf,x_expcdf]=cdfcalc(x);

yCDF = [];
xCDF = [];
x = x(~isnan(x));
n = length(x);
x = sort(x(:));
% Get cumulative sums
yCDF = (1:n)' / n;
% Remove duplicates; only need final one with total count
notdup = ([diff(x(:)); 1] > 0);
x_expcdf = x(notdup);
y_expcdf = [0; yCDF(notdup)];

%
% The theoretical CDF (theocdf) is assumed to be normal
% with unknown mean and sigma

zScores  =  (x_expcdf - mean(x))./std(x);

%theocdf  =  normcdf(zScores , 0 , 1);
mu = 0;
sigma = 1;
theocdf = 0.5 * erfc(-(zScores-mu)./(sqrt(2)*sigma));


%
% Compute the Maximum distance: max|S(x) - theocdf(x)|.
%

delta1    =  y_expcdf(1:end-1) - theocdf;   % Vertical difference at jumps approaching from the LEFT.
delta2    =  y_expcdf(2:end)   - theocdf;   % Vertical difference at jumps approaching from the RIGHT.
deltacdf  =  abs([delta1 ; delta2]);

KSmax =  max(deltacdf);
```

```{code-cell} ipython3
ks = []
for idx, feature in enumerate(features.rollaxis(1)):
    std_feature = np.std(feature)b
    mean_feature = np.mean(feature)
    thr_dist = std_feature * 3;
    thr_dist_min = mean_feature - thr_dist;
    thr_dist_max = mean_feature + thr_dist;
    aux = feature[np.logical_and(feature>thr_dist_min, feature<thr_dist_max)];

    if length(aux) > 10:
        ks.append(test_ks(aux))
    else:
        ks.append(0)
```

```{code-cell} ipython3
[A,ind] = sort(ks);
A = A(length(A)-par.max_inputs+1:end);
ncoeff = length(A);
maxA = max(A);
nd = 10;
d = (A(nd:end)-A(1:end-nd+1))/maxA*ncoeff/nd;
all_above1 = find(d>=1);
if numel(all_above1) >=2
    %temp_bla = smooth(diff(all_above1),3);
    aux2 = diff(all_above1);
    temp_bla = conv(aux2(:),[1 1 1]/3);
    temp_bla = temp_bla(2:end-1);
    temp_bla(1) = aux2(1);
    temp_bla(end) = aux2(end);

    thr_knee_diff = all_above1(find(temp_bla(2:end)==1,1))+(nd/2); %ask to be above 1 for 3 consecutive coefficients
    inputs = par.max_inputs-thr_knee_diff+1;
else
    inputs = par.min_inputs;
end


if  isfield(par,'plot_feature_stats') && par.plot_feature_stats
    [path,name,ext] = fileparts(par.filename);
    if isempty(path)
        path='.';
    end
    fig = figure('visible','off');
    stairs(sort(ks))
    hold on
    ylabel('ks_stat','interpreter','none')
    xlabel('#features')
            if ~isempty(inputs)
                line([numel(ks)-inputs+1 numel(ks)-inputs+1],ylim,'color','r')
            end
    line([numel(ks)-par.max_inputs numel(ks)-par.max_inputs],ylim,'LineStyle','--','color','k')
    title(sprintf('%s \n number of spikes = %d.  inputs_selected = %d.',name,number_of_spikes,inputs),'interpreter','none');
    print(fig,'-dpng',[path filesep 'feature_select_' name '.png'])
    close(fig)
end

if inputs > par.max_inputs
    inputs = par.max_inputs;
elseif isempty(inputs) || inputs < par.min_inputs
    inputs = par.min_inputs;
end

coeff(1:inputs)=ind(lengths:-1:lengths-inputs+1);


%CREATES INPUT MATRIX FOR SPC
input_for_spc  =zeros(number_of_spikes,inputs);
for i=1:number_of_spikes
    for j=1:inputs
        input_for_spc(i,j)=features(i,coeff(j));
    end
end
```
