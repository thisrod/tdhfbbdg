# Package compiler script for a Julia image for plotting

C = NaN
N = 10
l = 20.0
h = min(l/(N+1), sqrt(√2*π/N))
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
φ = @.cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
φ .*= z

include("figs.jl")

function plot_trace(z, col)
    plot!(real.(z), imag.(z), lc=col, leg=:none)
    scatter!(real.(z[1:1]), imag.(z[1:1]), ms=5, mc=col, msw=0)
end

poles(u) = (u,u)

widir(φ);
plot(sense_portrait(real(φ)) |> implot, aspect_ratio=1; imopts...);
plot_trace([-1im,1], :white);
plot(zplot(φ), xshowaxis=false, yshowaxis=false);
annotate!([1], [0], text("foo", :white, 9));
PE = plot()
scatter!(PE, [0.0, 1.0], [0.0, 1.0]; bpsty..., sqopts...)
plot!(PE, [0.0, 1.0], [0.0, 1.0]; insty...)
scatter!([0.0, 1.0], [0.0, 1.0])
plot!([0.0, 1.0], [0.0, 1.0])
xlims!(0,1)
ylims!(0,1)
xticks!(0:0.2:1.2)
yticks!(0:5:30)