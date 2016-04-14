-- Autograd
local autograd = require 'autograd'

-- Perturbation (finite diffs):
local perturbation = 1e-2

-- Threshold:
local threshold = 1e-4

-- Find grad:
local function findGrad(ref, x, dst)
   for k,v in pairs(ref) do
      if v == x then
         return dst[k]
      elseif type(v) == 'table' then
         local res = findGrad(ref[k], x, dst[k])
         if res then return res end
      end
   end
end

-- Compute grads with bprop:
local function jacobianFromAutograd(func, inputs, var)
   -- Autograd:
   local df = autograd(func)
   local grads = df(table.unpack(inputs))
   local gradsVerify = df(table.unpack(inputs))

   -- Find grad:
   local g = findGrad(inputs[1], var, grads)
   local gVerify = findGrad(inputs[1], var, gradsVerify)
   local err = (g - gVerify):abs():max()

   if err ~= 0 then
      error("autograd gradient not deterministic")
   end

   -- Return grads:
   return g:contiguous():view(-1):clone()
end

-- Compute grads from finite differences
local function jacobianFromFiniteDifferences(func, inputs, var)
   -- Flat view:
   local view = var:view(-1)

   -- Grads:
   local grads = view:clone():zero()

   -- Finite diffs:
   for i = 1,view:size(1) do
      -- Initial val:
      local val = view[i]

      -- Perturbate:
      view[i] = val - perturbation/2
      local pred1 = func(table.unpack(inputs))
      view[i] = val + perturbation/2
      local pred2 = func(table.unpack(inputs))
      view[i] = val

      -- Finite diff:
      grads[i] = (pred2-pred1) / perturbation
   end
   -- Return grads:
   return grads
end

local function gradcheckvar(func, inputs, var, randomizeInput)
   -- Random input:
   if randomizeInput then
      var:uniform(-10,10)
   end

   -- Estimate grads with fprop:
   local jacobian = jacobianFromAutograd(func, inputs, var)

   local originalLoss = func(table.unpack(inputs))

   local noise = jacobian:view(-1):clone():zero()

   local idx = math.random(1, noise:size(1))

   noise:narrow(1,idx,1):uniform(-perturbation, perturbation)

   local varBackup = var:clone()

   var:add(torch.view(noise, var:size()))

   local perturbedLoss = func(table.unpack(inputs))

   local approxPerturbed = originalLoss + torch.dot(jacobian, noise)

   -- Error:
   local err = math.abs((perturbedLoss - approxPerturbed)) / (math.max(perturbedLoss, originalLoss))

   -- Threhold?
   local pass = err < threshold
   if not pass then
      print('original loss = '..originalLoss)
      print('perturbed loss = '..perturbedLoss)
      print('approximated perturbed loss = '..approxPerturbed)
      print('error = ' .. err)
   end
   return pass
end

-- Test grads:
return function(opt)
   -- Options
   local randomizeInput = opt.randomizeInput
   if randomizeInput == nil then
      randomizeInput = true
   end

   -- Run grad check:
   local function gradcheck(func, ...)
      local args = {...}
      -- get all vars:
      local vars = autograd.util.sortedFlatten(args[1])
      local ok = true
      for i,var in ipairs(vars) do
         ok = ok and gradcheckvar(func, args, var, randomizeInput)
      end
      return ok
   end

   -- Grad check fun:
   return gradcheck
end
