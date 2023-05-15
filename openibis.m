function depthOfAnesthesia = openibis(eeg)   % The input eeg is a column vector, units of uV.  Outputs one score per EEG epoch.
[Fs,stride]		= deal(128,0.5); % The EEG sampling frequency Fs must be 128 Hz, and each EEG window advances 0.5 seconds.
[BSRmap,BSR] 	= suppression(eeg,Fs,stride);  % Identify burst suppressed segments and calculate Burst Suppression Rates.
components = logPowerRatios(eeg,Fs,stride,BSRmap);  % Calculate the other components for the depth-of-anesthesia.
depthOfAnesthesia = mixer(components,BSR); % Mix the components and BSR together to generate the depth-of-anesthesia scores.

function [BSRmap,BSR] = suppression(eeg,Fs,stride) % Determines whether an EEG segment is burst-suppressed and calculates BSR.
[N,nStride]	= nEpochs(eeg,Fs,stride); 	 % Calculate the total number of epochs N and the number of samples per stride.
BSRmap		= zeros(N,1);  % Allocate space to store a true/false map of whether epochs are burst suppressed.
for n = 1:N		% Evaluate over the available epochs.
	x 	= segment(eeg,n+6.5,2,nStride);	% Obtain two half-second strides (1 second).
 	BSRmap(n)  = all(abs(x -baseline(x)) <= 5); % Define this epoch as suppressed if all samples are within 5 uV of baseline.
end
BSR  = 100 * movmean(BSRmap,[(63/stride)-1,0]); % BSR is the percentage of burst suppression in the previous 63 sec.

function components = logPowerRatios(eeg,Fs,stride,BSRmap)  % Calculates a family of components, based on log-power ratios.
[N,nStride]	= nEpochs(eeg,Fs,stride); 	% Calculate the total number of epochs and the number of samples per stride.
[B,A]		= butter(2,0.65/(Fs/2),'high');	% Create a second order high-pass Butterworth filter at 0.65 Hz.
eegHiPassFiltered  	= filter(B,A,eeg);		% ... and filter out very low frequencies from the input EEG.
psd		= nan(N,4*nStride/2);		% Allocate space to store power spectral densities, 0 to 63.5 Hz with 0.5 Hz bins.
SuppressionFilter	= piecewise(0:0.5:63.5,[0,3,6],[0,0.25,1]).^2;  % Define a suppression filter in the range of 0 - 6 Hz
components 	= nan(N,3);		% Allocate space to store output signal components, 3 components for each epoch
for n = 1:N  	 % Evaluate over the available epochs. Epochs are 2 seconds long,  and the stride is 0.5 s, hence 75% overlap
    if isNotBurstSuppressed(BSRmap,n,4) 	% If this EEG epoch  (4 strides = 2 seconds) does not contain any burst suppression                                                         ...
	    psd(n,:) = powerSpectralDensity(segment(eegHiPassFiltered,n+4,4,nStride));  % then calculate Power Spectral Densities.
        if sawtoothDetector(segment(eeg,n+4,4,nStride),nStride) % If sawtooth-shaped K-complexes are detected
            psd(n,:)  = SuppressionFilter .*  psd(n,:);    % then suppress low frequencies accordingly.
        end
    end
    thirtySec	= timeRange(30,n,stride); 	% Consider data from the most recent thirty seconds.
    VhighPowerConc  = sqrt(mean(psd(thirtySec,bandRange(39.5,46.5,0.5)).*psd(thirtySec,bandRange(40,47,0.5)),2));
    wholePowerConc  = sqrt(mean(psd(thirtySec,bandRange(  0.5,46.5,0.5)).*psd(thirtySec,bandRange( 1,47,0.5)),2));
    midBandPower  = prctmean(nanmean(10*log10(psd(thirtySec,bandRange(11,20,0.5))),1),50,100);% 11 - 20Hz band power in dB.
    %% 
    
    components(n,1) = meanBandPower(psd(thirtySec,:),30,47,0.5) - midBandPower;  % Used for sedation depth-of-anesthesia.
    components(n,2) = trimmean(10*log10(VhighPowerConc./wholePowerConc),50);  % Used for general depth-of-anesthesia.
    components(n,3) = meanBandPower(psd(thirtySec,:),0.5,4,0.5)  - midBandPower;  % For weighting between sedation and general.
end

function y = powerSpectralDensity(x)  % Calculates the Blackman-windowed Power Spectral Density.
f = fft(blackman(length(x)) .* (x - baseline(x)));% Perform a Blackman-windowed FFT,removing baseline signal drift.
y = 2*abs(f(1:length(x)/2)').^2 /(length(x)*sum(blackman(length(x)).^2));  % Convert frequency amplitudes to frequency powers.

function y = sawtoothDetector(eeg,nStride) % Determines if this EEG segment contains a strong sawtooth-shaped K-complex.
saw = [zeros(1,nStride-5),1:5]'; saw = (saw-mean(saw))/std(saw,1); % Construct a normalized sawtoothed waveform.
r = 1:(length(eeg)-length(saw));
v = movvar(eeg,[0 length(saw)-1],1);
m = ([conv(eeg,flipud(saw),'valid') conv(eeg,saw,'valid')]/length(saw)).^2; % Match the sawtooth to the EEG by convolution.
y = max([(v(r)>10).*m(r,1)./v(r); (v(r)>10).*m(r,2)./v(r)]) > 0.63;  % Return a true value if there are any strong matches.


function y = mixer(components,BSR) % Generates the output depth-of-anesthesia by converting and weighting components,                                                                                                                      BSRs.
sedationScore = scurve(components(:,1),104.4,49.4,-13.9,5.29);  % Map component 1 to a sedation score on a logistic S-curve.
generalScore = piecewise(components(:,2),[-60.89,-30],[-40,43.1]);  % Map component 2 to a general score,  linear region                        
generalScore = generalScore + scurve(components(:,2),61.3,72.6,-24.0,3.55).*(components(:,2)>=-30);  % and S-curved region.
bsrScore = piecewise(BSR,[0,100],[50,0]); % Convert the BSR to a BSR score using a piecewise linear function.
generalWeight = piecewise(components(:,3),[0,5],[0.5,1]) .* (generalScore<sedationScore);  % Convert component 3 to a weight                 
bsrWeight  = piecewise(BSR,[10,50],[0,1]);  % Convert the BSR to a weight.
x = (sedationScore .* (1-generalWeight)) + (generalScore .* generalWeight); % Weight the sedation and general scores together.
y = piecewise(x,[-40,10,97,110],[0,10,97,100]).*(1-bsrWeight) + bsrScore.*bsrWeight; % Compress and weight these with the BSR.

% The remaining functions are the same as before, with no errors found.
function [N,nStride]  = nEpochs(eeg,Fs,stride), nStride = Fs*stride; N=floor((length(eeg)-Fs)/nStride)-10; % Number of epochs.
function y = meanBandPower(psd,from,to,bins),   v = psd(:,bandRange(from,to,bins));  y = mean(10*log10(v(~isnan(v))));  % in dB.
function y = bandRange(from,to,bins),           y = ((from/bins):(to/bins))+1; % Indices of bins for a given frequency band.
function y = baseline(x),                       v = (1:length(x))'.^(0:1); y = v * (v\x); % Fits a baseline to the input.
function y = bound(x,lowerBound,upperBound),    y = min(max(x,lowerBound),upperBound); % Fixes the input between bounds.
function y = segment(eeg,from,number,nStride),  y = eeg(from*nStride+(1:number*nStride)); % Extracts a segment of the EEG data.
function y = isNotBurstSuppressed(BSRmap,n,p),  y = ~((n<p)|| any(BSRmap(n+((1-p):0))));  % Checks if not burst suppressed.
function y = timeRange(seconds,n,stride),       y = max(1,n-(seconds/stride)+1):n;  % Indices for the most recent time points.
function y = prctmean(x,lo,hi),                 v = prctile(x,[lo hi]);   y = mean(x(x>=v(1) & x<=v(2)));  % Percentile-band mean.
function y = piecewise(x,xp,yp),                y = interp1(xp,yp,bound(x,xp(1),xp(end)));   % Response on a piecewise-linear fn.
function y = scurve(x,Eo,Emax,x50,xwidth),      y = Eo -  Emax./(1+exp((x-x50)/xwidth)); % Response on a logistic S-curve fn.
