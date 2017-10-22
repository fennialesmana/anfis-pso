function [time, Et, iserror]=ANFISPSO()
tic;
clc;
clear;
close all;

% load dataset
data = csvread('seeds.csv');
input_data = data(:, 1:end-1);
output_data = data(:, end);

% Parameter initialization
[center,U] = fcm(input_data, 3, [2 100 1e-6]); %center = center cluster, U = membership level
[total_examples, total_features] = size(input_data);
class = 3; % [Changeable]
epoch = 0;
epochmax = 400; % [Changeable]
%Et = zeros(epochmax, 1);
[Yy, Ii] = max(U); % Yy = max value between both membership function, Ii = the class corresponding to the max value

% Population initialization
pop_size = 10;
premise_population = zeros(pop_size, 3, class, total_features); % parameter: population size * 6 * total classes * total features
consequent_population = rand(pop_size, (total_features+1)*class, 1);

% Velocity
premise_velocity = zeros(pop_size, 3, class, total_features); % velocity matrix of an iteration
consequent_velocity = zeros(pop_size, (total_features+1)*class, 1);

c1 = 1.2;
c2 = 1.2;
r1 = 0.4;
r2 = 0.6;
counter = 0;
iserror = 0;
for particle=1:pop_size
    a = zeros(class, total_features);
    b = repmat(2, class, total_features);
    c = zeros(class, total_features);

    for k =1:class
        for i = 1:total_features % looping for all features
            % premise parameter: a
            aTemp = (max(input_data(:, i))-min(input_data(:, i)))/(2*sum(Ii' == k)-2);
            aLower = aTemp*0.5;
            aUpper = aTemp*1.5;
            a(k, i) = (aUpper-aLower).*rand()+aLower;

            %premise parameter: c
            dcc = (2.1-1.9).*rand()+1.9;
            cLower = center(k,total_features)-dcc/2;
            cUpper = center(k,total_features)+dcc/2;
            c(k,i) = (cUpper-cLower).*rand()+cLower;
        end
    end

    premise_population(particle, 1, :, :) = a;
    premise_population(particle, 2, :, :) = b;
    premise_population(particle, 3, :, :) = c;
end

% pBest initialization
pBest_fitness = repmat(10000, pop_size, 1);
pBest_premise_position = zeros(pop_size, 3, class, total_features);
pBest_consequent_position = zeros(pop_size, (total_features+1)*class, 1);

% calculate fitness function
for i=1:pop_size
    premise_particle_position = squeeze(premise_population(i, :, :, :));
    consequent_particle_position = squeeze(consequent_population(i, :, :));
    e = get_fitness(premise_particle_position, consequent_particle_position, class, input_data, output_data);
    if isreal(e) == 0
        Et = [];
        time = 0;
        iserror = 1;
        return;
    end
    if e < pBest_fitness(i)
        pBest_fitness(i) = e;
        pBest_premise_position(i, :, :, :) = premise_particle_position;
        pBest_consequent_position(i, :, :) = consequent_particle_position;
    end
end

% find gBest
[gBest_fitness, idx] = min(pBest_fitness);
gBest_premise_position = squeeze(pBest_premise_position(idx, :, :, :));
gBest_consequent_position = squeeze(pBest_consequent_position(idx, :, :));

% ITERATION
while epoch < epochmax
    epoch = epoch + 1;

    % calculate velocity and update particle
    % vi(t + 1) = wvi(t) + c1r1(pbi(t) - pi(t)) + c2r2(pg(t) - pi(t))
    % pi(t + 1) = pi(t) + vi(t + 1)
    r1 = rand();
    r2 = rand();
    for i=1:pop_size
        premise_velocity(i, :, :, :) = squeeze(premise_velocity(i, :, :, :)) + ((c1 * r1) .* (squeeze(pBest_premise_position(i, :, :, :)) - squeeze(premise_population(i, :, :, :)))) + ((c2 * r2) .* (gBest_premise_position(:, :, :) - squeeze(premise_population(i, :, :, :))));
        consequent_velocity(i, :, :) = squeeze(consequent_velocity(i, :, :)) + ((c1 * r1) .* (squeeze(pBest_consequent_position(i, :, :)) - squeeze(consequent_population(i, :, :)))) + ((c2 * r2) .* (gBest_consequent_position(:, :) - squeeze(consequent_population(i, :, :))));
        premise_population(i, :, :, :) = premise_population(i, :, :, :) + premise_velocity(i ,:, :, :);
        consequent_population(i, :, :) = consequent_population(i, :, :) + consequent_velocity(i ,:, :);
    end
    
    % calculate fitness value and update pBest
    for i=1:pop_size
        premise_particle_position = squeeze(premise_population(i, :, :, :));
        consequent_particle_position = squeeze(consequent_population(i, :, :));
        e = get_fitness(premise_particle_position, consequent_particle_position, class, input_data, output_data);
        if isreal(e) == 0
            Et = [];
            time = 0;
            iserror = 1;
            return;
        end        
        if e < pBest_fitness(i)
            pBest_fitness(i) = e;
            pBest_premise_position(i, :, :, :) = premise_particle_position;
            pBest_consequent_position(i, :, :) = consequent_particle_position;
        end
    end
    
    % find gBest
    [gBest_fitness, idx] = min(pBest_fitness);
    gBest_premise_position = squeeze(pBest_premise_position(idx, :, :, :));
    gBest_consequent_position = squeeze(pBest_consequent_position(idx, :, :));    
    
    Et(epoch) = gBest_fitness;

    % Draw the SSE plot
%     plot(1:epoch, Et);
%     title(['Epoch  ' int2str(epoch) ' -> MSE = ' num2str(Et(epoch))]);
%     grid
%     pause(0.001);
end

%[out output out-output]
% ----------------------------------------------------------------
time = toc;
end