% function y=synae_prova2()
clear all; clc;
[x,fs]=wavread('x1');% x=randn(1,fs*Tdur);
Tdur=1.5; %sec
fs=20e3;
x=x(end-1:-1:end-Tdur*fs).*hanning(Tdur*fs);
L = length( x );            % length of the speech signal

Tw = 32;    % analysis frame duration (ms) 
Ts = Tw/8;  % analysis frame shift (ms)

Nw = round( fs*Tw*0.001 );  % frame duration (in samples)
Ns = round( fs*Ts*0.001 );  % frame shift (in samples)
nfft = 2^nextpow2( 2*Nw );  % FFT analysis length


% divide noisy and clean speech signals into frames
[ frames.x, indexes ] = vec2frames( x, Nw, Ns, 'rows', @hanning, true );

% perform short-time Fourier transform (STFT) analyses
spectrum.X = fft( frames.x, nfft, 2 );           

% get Gains from Image
I=imread('C:\Users\joeDiHare\Pictures\io_BW.png');
I2=double(I(:,:,1))./255;
I2r=imresize(I2',[size(spectrum.X,1) size(spectrum.X,2)/2]);%imshow(I2)
MASK=[I2r(:,end:-1:1) I2r];
%MASK=[I2r(end:-1:1,:); I2r(1:1:end,:)];

% apply the ideal binary mask and create modified complex spectrum
spectrum.Y = MASK .* exp(1i*angle(spectrum.X));%abs(spectrum.X) .* 

% alternatively, you could perform a sanity check and reconstruct the clean speech...
% spectrum.Y = abs(spectrum.S) .* exp(j*angle(spectrum.S));

% apply inverse STFT 
frames.y = real( ifft(spectrum.Y,nfft,2) ); 

% discard FFT padding from frames
frames.y = frames.y(:, 1:Nw);

% perform overlap-and-add synthesis
y = frames2vec( frames.y, indexes, 'rows', @hanning, 'G&L' ); 

% truncate extra padding (back to original signal length)
y = y(1:L);

sound(y,fs)
%
t=linspace(0,Tdur,Tdur*fs);
figure(1); 
subplot(121); plot(t,x,'k',t,y,'r'); ylim([-1.5 1.5])
% subplot(121); imshow(MASK);
subplot(122); spectrogram(y,4*Tw,Tw,nfft,fs,'yaxis');
%% 
% end
% 
% function [ frames, indexes ] = vec2frames( vec, Nw, Ns, direction, window, padding )
% % VEC2FRAMES Splits signal into overlapped frames using indexing.
% % 
% %   B=vec2frames(A,M,N) creates a matrix B whose columns consist of 
% %   segments of length M, taken at every N samples along input vector A.
% %
% %   [B,R]=vec2frames(A,M,N,D,W,P) creates a matrix B whose columns 
% %   or rows, as specified by D, consist of segments of length M, taken 
% %   at every N samples along the input vector A and windowed using the
% %   analysis window specified by W. The division of A into frames is 
% %   achieved using indexes returned in R as follows: B=A(R);
% %
% %   Summary
% %
% %           A is an input vector
% %
% %           M is a frame length (in samples)
% %
% %           N is a frame shift (in samples)
% %
% %           D specifies if the frames in B are rows or columns,
% %             i.e., D = 'rows' or 'cols', respectively
% %
% %           W is an optional analysis window function to be applied to 
% %             each frame, given as a function handle, e.g., W = @hanning
% %             or as a vector of window samples, e.g., W = hanning( M )
% %
% %           P specifies if last frame should be padded to full length,
% %             or simply discarded, i.e., P = true or false, respectively
% %   
% %           B is the output matrix of frames
% %
% %           R is a matrix of indexes used for framing, such that division 
% %             of A into frames is achieved as follows: B=A(R);
% %
% %   Examples
% %
% %           % divide the input vector into seven-sample-long frames with a shift
% %           % of three samples and return frames as columns of the output matrix
% %           % (note that the last sample of the input vector is discarded)
% %           vec2frames( [1:20], 7, 3 )
% %
% %           % divide the input vector into seven-sample-long frames with a shift
% %           % of three samples and return frames as rows of the output matrix
% %           % (note that the last sample of the input vector is discarded)
% %           vec2frames( [1:20], 7, 3, 'rows' )
% %
% %           % divide the input vector into seven-sample-long frames with a shift
% %           % of three samples, pad the last frame with zeros so that no samples
% %           % are discarded and return frames as rows of the output matrix
% %           vec2frames( [1:20], 7, 3, 'rows', [], true )
% %
% %           % divide the input vector into seven-sample-long frames with a shift
% %           % of three samples, pad the last frame with white Gaussian noise
% %           % of variance (1E-5)^2 so that no samples are discarded and 
% %           % return frames as rows of the output matrix
% %           vec2frames( [1:20], 7, 3, 'rows', false, { 'noise', 1E-5 } )
% %
% %           % divide the input vector into seven-sample-long frames with a shift
% %           % of three samples, pad the last frame with zeros so that no samples 
% %           % are discarded, apply the Hanning analysis window to each frame and
% %           % return frames as columns of the output matrix
% %           vec2frames( [1:20], 7, 3, 'cols', @hanning, 0 )
% % 
% %   See also FRAMES2VEC, DEMO
% 
% %   Author: Kamil Wojcicki, UTD, July 2011
% 
% 
%     % usage information
%     usage = 'usage: [ frames, indexes ] = vec2frames( vector, frame_length, frame_shift, direction, window, padding );';
% 
%     % default settings 
%     switch( nargin )
%     case { 0, 1, 2 }, error( usage );
%     case 3, padding=false; window=false; direction='cols';
%     case 4, padding=false; window=false; 
%     case 5, padding=false; 
%     end
% 
%     % input validation
%     if( isempty(vec) || isempty(Nw) || isempty(Ns) ), error( usage ); end;
%     if( min(size(vec))~=1 ), error( usage ); end;
%     if( Nw==0 || Ns==0 ), error( usage ); end;
% 
%     vec = vec(:);                       % ensure column vector
% 
%     L = length( vec );                  % length of the input vector
%     M = floor((L-Nw)/Ns+1);             % number of frames 
% 
% 
%     % perform signal padding to enable exact division of signal samples into frames 
%     % (note that if padding is disabled, some samples may be discarded)
%     if( ~isempty(padding) )
%  
%         % figure out if the input vector can be divided into frames exactly
%         E = (L-((M-1)*Ns+Nw));
% 
%         % see if padding is actually needed
%         if( E>0 ) 
% 
%             % how much padding will be needed to complete the last frame?
%             P = Nw-E;
% 
%             % pad with zeros
%             if( islogical(padding) && padding ) 
%                 vec = [ vec; zeros(P,1) ];
% 
%             % pad with a specific numeric constant
%             elseif( isnumeric(padding) && length(padding)==1 ) 
%                 vec = [ vec; padding*ones(P,1) ];
% 
%             % pad with a low variance white Gaussian noise
%             elseif( isstr(padding) && strcmp(padding,'noise') ) 
%                 vec = [ vec; 1E-6*randn(P,1) ];
% 
%             % pad with a specific variance white Gaussian noise
%             elseif( iscell(padding) && strcmp(padding{1},'noise') ) 
%                 if( length(padding)>1 ), scale = padding{2}; 
%                 else, scale = 1E-6; end;
%                 vec = [ vec; scale*randn(P,1) ];
% 
%             % if not padding required, decrement frame count
%             % (not very elegant solution)
%             else
%                 M = M-1;
% 
%             end
% 
%             % increment the frame count
%             M = M+1;
%         end
%     end
% 
% 
%     % compute index matrix 
%     switch( direction )
% 
%     case 'rows'                                                 % for frames as rows
%         indf = Ns*[ 0:(M-1) ].';                                % indexes for frames      
%         inds = [ 1:Nw ];                                        % indexes for samples
%         indexes = indf(:,ones(1,Nw)) + inds(ones(M,1),:);       % combined framing indexes
%     
%     case 'cols'                                                 % for frames as columns
%         indf = Ns*[ 0:(M-1) ];                                  % indexes for frames      
%         inds = [ 1:Nw ].';                                      % indexes for samples
%         indexes = indf(ones(Nw,1),:) + inds(:,ones(1,M));       % combined framing indexes
%     
%     otherwise
%         error( sprintf('Direction: %s not supported!\n', direction) ); 
% 
%     end
% 
% 
%     % divide the input signal into frames using indexing
%     frames = vec( indexes );
% 
% 
%     % return if custom analysis windowing was not requested
%     if( isempty(window) || ( islogical(window) && ~window ) ), return; end;
%     
%     % if analysis window function handle was specified, generate window samples
%     if( isa(window,'function_handle') )
%         window = window( Nw );
%     end
%     
%     % make sure analysis window is numeric and of correct length, otherwise return
%     if( isnumeric(window) && length(window)==Nw )
% 
%         % apply analysis windowing beyond the implicit rectangular window function
%         switch( direction )
%         case 'rows', frames = frames * diag( window );
%         case 'cols', frames = diag( window ) * frames;
%         end
% 
%     end
% 
% 
% % EOF 
% end