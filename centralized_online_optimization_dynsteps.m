function [wc]=centralized_online_optimization_dynsteps(T,D,G,verbose,compute_step_size)
    % Parameters
    % param.mu = 0.5;
    param_F.R = G; % /!\ param.R represents the radius over the norm of G and not the radius of the domain /!\
    function_class = 'ConvexBoundedGradient';
    % END - Parameters
    indices = cell(1,T);
    % PEP stuff
    P = pep();
    
    % Declaring multiple functions
    multiFunctions = @(x) P.DeclareFunction(function_class, param_F);
    F = foreach(multiFunctions,indices);
    
    % Declaring an indicator function
    param_id.D = D;
    id = P.DeclareFunction('ConvexIndicator',param_id);

    % Computing the reference point----------------
    % Aggregating functions together
    F_sum = F{1}; % Initializing the sum
    for t = 2:T
        F_sum = F_sum + F{t};
    end
    F_sum = F_sum + id; % Adding the indicator function
    
    [xs,fs]=F_sum.OptimalPoint(); % Computing the optima
    %----------------------------------------------
    
    % Declaring a starting point
    x1=P.StartingPoint();
    P.InitialCondition((x1-xs)^2<=D^2); % Add an initial condition ||x0-xs||^2<= 1

    % (3) Algorithm
    x=cell(T+1,1);% we store the iterates in a cell for convenience
    y=cell(T,1);
    x{1}= x1;
    for t=1:T
        y{t} = x{t} - compute_step_size(t,D,G)* F{t}.gradient(x{t}); % Propagating 
        x{t+1} = projection_step(y{t},id);
    end

    % (4) Set up the performance measure
    % We fix the value F_t(x_t)
    F_fixed = cell(T,1);
    G_fixed = cell(T,1);
    F_sum_fixed = 0;
    for t=1:T % the oracle is called on 'F_t(x_t)' for all t from 1 to T
        %BEFORE: [g,f]=F{t}.oracle(x{t+1}); % g=grad F(x), f=F(x)
        [g,f]=F{t}.oracle(x{t}); % g=grad F(x), f=F(x)
        id_val = id.value(x{t});
        % To be used into the Performance Metric
        F_fixed{t}=f;
        G_fixed{t}=g;
        F_sum_fixed=F_sum_fixed + f + id_val;        
        % - - - 
    end
    
    P.PerformanceMetric(F_sum_fixed - fs); % Worst-case evaluated as F(x)-F(xs)
    
    options = sdpsettings('verbose',max(verbose-1,0));
    out=P.solve(verbose, options);

    % Post-treatment 
    if contains(out.solverDetails.info,"Successfully solved")
        wc=out.WCperformance;
    else
        if contains(out.solverDetails.info,"Unbounded objective")
            wc=-1;
        else
            wc=-2; % Other types of errors
        end
    end

    fprintf('PESTO bound: %d \n',wc);
end

