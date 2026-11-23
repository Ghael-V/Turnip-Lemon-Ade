@localsize 8, 1, 1
@buf 8 1, 2, 3, 4, 5, 6, 7, 8
@invocationid(r0.x)

; Create the 64bit byte offset for this invocation in r2.xy
shl.b r2.x, r0.x, 2
mov.u32u32 r2.y, 0

(rpt5)nop
ldgb.untyped.4d.u32.4 r1.x, g[0], r2.x, r0.x
(sy)add.u r1.x, r1.x, 5
(rpt5)nop
stgb.untyped.4d.u32.4 g[0], r1.x, r0.x, r2.x
end
