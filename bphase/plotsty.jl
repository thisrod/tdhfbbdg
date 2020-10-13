Superfluids.default!(:xlims, (-5,5))
Superfluids.default!(:ylims, (-5,5))
Plots.default(:legend, :none)
Plots.default(:dpi, 72)
Plots.default(:leg, :none)
Plots.default(:framestyle, :box)
Plots.default(:fontfamily, "Latin Modern Sans")
Plots.default(:size, (200,200))

function plot_trace(z, col)
    plot!(real.(z), imag.(z), lc=col, leg=:none)
    scatter!(real.(z[1:1]), imag.(z[1:1]), ms=5, mc=col, msw=0)
end

# Colors for Berry phase plots
snapsty = RGB(0.3,0,0)
bpsty, impsty, insty, outsty, nsty =
    distinguishable_colors(5+3, [snapsty, RGB(1,1,1), RGB(0,0,0)])[4:end]
bpsty = (ms=2, mc=bpsty, msc=0.5bpsty)
impsty = (ms=2, mc=impsty, msc=0.5impsty)
insty = (lc=insty,)
outsty = (lc=outsty,)
nsty = (lc=nsty,)
snapsty = (lc=snapsty, lw=0.5)

recsize=(100,200)
