using Plots, ComplexPhasePortrait, JLD2, Plots.PlotMeasures

# Plots and portraits are a bit wierd with pixel arrays
implot(x,y,image) = plot(x, y, image,
    yflip=false, aspect_ratio=1, framestyle=:box, tick_direction=:out)
implot(image) = implot(y,y,image)
saneportrait(u) = reverse(portrait(u), dims=1)
zplot(u) = implot(saneportrait(u).*abs2.(u)/maximum(abs2,u))
argplot(u) = implot(saneportrait(u))
zplot(u::Matrix{<:Real}) = zplot(u .|> Complex)
argplot(u::Matrix{<:Real}) = argplot(u .|> Complex)

N = 100
l = 20.0	# maximum domain size

h = min(l/(N+1), sqrt(√2*π/N))
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y

y = [-10, 10]	# Plots.jl pixel offset

@load "ss.jld2" S1t zz1 q1 zz2 q2 bp1 bp2 ap1 ap2 wp1 wp2

xlab = "x (SHO length)"
ylab = "y (SHO length)"

function lbl!(s)
    annotate!(-5, 5, Plots.text("($s)", :left, :top, :white))
    savefig(P, "../figs/resp200702$(s).pdf")
end

popts = (xlims=(-5,5), ylims=(-5,5),
    fontfamily="Latin Modern Sans")

P = plot(zplot(q2), xshowaxis=false,
    size=(303,300), bottom_margin=-6mm, left_margin=-3mm; popts...)
plot!(real.(zz1), imag.(zz1), lc=:white, leg=:none)
scatter!(real.(zz1[1:1]), imag.(zz1[1:1]), ms=5, mc=:white, msw=0)
lbl!('a')

P = plot(zplot(q1), xshowaxis=false, yshowaxis=false,
    size=(300,292), bottom_margin=-6mm, left_margin=-6mm; popts...)
plot!(real.(zz2), imag.(zz2), lc=:white, leg=:none)
scatter!(real.(zz2[1:1]), imag.(zz2[1:1]), ms=5, mc=:white, msw=0)
lbl!('b')

P = plot(zplot(bp2), size=(300,292), bottom_margin=-3mm, left_margin=-3mm; popts...)
lbl!('c')

P = plot(zplot(bp1), yshowaxis=false, size=(303,300), bottom_margin=-3mm, left_margin=-6mm; popts...)
lbl!('d')

P = scatter(S1t, ap1./2π, label="GPE Berry",
    leg=:topleft, size=(600,292), framestyle=:box,
    fontfamily="Latin Modern Sans", ms=3)
scatter!(S1t, wp1./2π, label="GPE Wu", ms=3)
scatter!(S1t, -ap2./2π, label="Imp. Berry", ms=3)
scatter!(S1t, wp2./2π, label="Imp. Wu", ms=3)
annotate!(xlims()[1], ylims()[end], Plots.text("(e)", :left, :top, :black))
savefig(P, "../figs/resp200702e.pdf")

`xetex figone.tex`