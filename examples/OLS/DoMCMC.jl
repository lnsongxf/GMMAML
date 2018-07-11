using Distributions
include("Model.jl") #
include(Pkg.dir()"/MPI/examples/montecarlo.jl")

function OLSWrapper()
    n = 500 # sample size
    outfile = "OLS-GMM.out"
    verbosity = true
    chain = 0.0 # initialize outside loop
    # get the data for the rep (drawn from design at true param. values)
    θinit = [zeros(11); 1.0]
    y, x, θtrue = dgp(n)
    # define things for MCMC
    burnin = 20000
    ChainLength = 20000
    lnL = θ -> logL(θ, y, x)
    # initial proposal moves one at a time
    Proposal = θ -> proposal1(θ, tuning)
    Prior = θ -> prior(θ) # uniform, doesn't matter
    tuning = 0.05*ones(size(θinit))
    chain = mcmc(θinit, ChainLength, burnin, Prior, lnL, Proposal, verbosity)
    θinit = vec(mean(chain[:,1:end-1],1))
    # keep every 10th
    i = 1:size(chain,1)
    keep = mod.(i,10.0).==0
    chain = chain[keep,:]
    # now use a MVN random walk proposal 
    Σ = cov(chain[:,1:end-1])
    tuning = 0.2
    for j = 1:5
        P = chol(Σ)
        Proposal = θ -> proposal2(θ,tuning*P)
        if j == 5
            ChainLength = 100000
        end    
        chain = mcmc(θinit, ChainLength, 0, Prior, lnL, Proposal, verbosity)
        accept = mean(chain[:,end])
        #println("tuning: ", tuning, "  acceptance rate: ", accept)
        if accept > 0.35
            tuning *= 1.2
        elseif accept < 0.25
            tuning *= 0.8
        end
        # keep every 10th
        i = 1:size(chain,1)
        keep = mod.(i,10.0).==0
        chain = chain[keep,:]
        θinit = vec(mean(chain[:,1:end-1],1))
        Σ = 0.5*Σ + 0.5*cov(chain[:,1:end-1])
    end
    # make the Zs for the thetas
    #option = "Asymptotic"
    option = "Bootstrap"
    junk = MakeZ(θinit, y, x, option) # get size 
    Z = zeros(size(chain,1), size(junk,2))
    S = size(chain,1)
    for i = 1:S
        θ = chain[i,1:end-1]
        Z[i,:] = MakeZ(θ, y, x, option)
    end
    dstats(chain);
    println(θtrue)
    return chain[:,1:end-1], Z
end    

# the monitoring function
function OLSMonitor(sofar, results)
    if mod(sofar,10) == 0
        θtrue = [quantile(Normal(),τ) 2.0]
        # MCMC fit
        m = mean(results[1:sofar,[1;2]],1)
        er = m - θtrue
        b = mean(er,1)
        s = std(results[1:sofar,[1;2]],1) 
        mse = s.^2 + b.^2
        rmse = sqrt.(mse)
        println()
        println("QIV Results")
        println("reps so far: ", sofar)
        println("Plain MCMC results")
        println("rmse: ",rmse)
        println()
        # local linear fit
        println("local linear results")
        for i = 1:10
            m = mean(results[1:sofar,i*2+1:i*2+2],1)
            er = m - θtrue
            b = mean(er,1)
            s = std(results[1:sofar,i*2+1:i*2+2],1) 
            mse = s.^2 + b.^2
            rmse = sqrt.(mse)
            println("rmse: ",rmse)
        end   
    end
    # save the results
    if sofar == size(results,1)
        writedlm(outfile, results)
    end
end

function main()
    MPI.Init()
    reps = 1000   # desired number of MC reps
    n_returns = 22
    pooled = 1  # do this many reps before reporting
    montecarlo(OLSWrapper, OLSMonitor, MPI.COMM_WORLD, reps, n_returns, pooled)
    MPI.Finalize()
end

r = OLSWrapper(); # use this for run without MPI
#R = zeros(1,size(r,1))
#R[1,:]=r
#QIVMonitor(1, R)
#main()
