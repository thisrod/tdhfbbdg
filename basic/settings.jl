# Plot residual and time step settings for given accuracy

using Plots, JLD2

s = ["0", "1", "2", "4"]
P1 = plot(xlabel="error", ylabel="residual", link=:x, xscale=:log10, yscale=:log10)
title!("Convergence settings for 60x60 grid")
P2 = plot(xlabel="error", ylabel="time step", link=:x, xscale=:log10, yscale=:log10)

for s = ["0", "1", "2", "4"]
    @load "cvg_$(s).jld2" ats aers dts ders
    
    scatter!(P1, ders, dts, label=s)
    scatter!(P2, aers, ats, label=s)
end

xlims!(-Inf, 1e-6)

plot(P1, P2, layout=@layout [a; b]) |> display
