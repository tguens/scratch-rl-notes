-- Build-time fixes. docs/*.md stay unchanged (they must render on GitHub too).
-- 1. Drop the first H1 of the page (the title comes from the .qmd YAML).
-- 2. Obsidian/GitHub alerts "> [!type] Title" -> bold blue text (class .alert-blue, see styles.css).
-- 3. Links _appendix.md -> appendix.html, main.md -> index.html.
-- 4. Links ../src/x -> GitHub blob URL (the code is not part of the site).
-- 5. Image paths ../figs/ -> figs/ (includes resolve from the .qmd directory).

local seen_h1 = false

function Header(el)
  if el.level == 1 and not seen_h1 then
    seen_h1 = true
    return {}
  end
end

local known = {
  note = true, tip = true, important = true, warning = true, caution = true,
}

function BlockQuote(el)
  local first = el.content[1]
  if not first or (first.t ~= "Para" and first.t ~= "Plain") then return nil end
  local tag = first.content[1]
  if not tag or tag.t ~= "Str" then return nil end
  local kind = tag.text:match("^%[!(%a+)%]$")
  if not kind then return nil end
  kind = kind:lower()
  if not known[kind] then return nil end

  -- Title = inlines after the tag, up to the first line break.
  local title, rest = {}, {}
  local in_title = true
  for i = 2, #first.content do
    local inl = first.content[i]
    if in_title and (inl.t == "SoftBreak" or inl.t == "LineBreak") then
      in_title = false
    elseif in_title then
      title[#title + 1] = inl
    else
      rest[#rest + 1] = inl
    end
  end
  while #title > 0 and title[1].t == "Space" do table.remove(title, 1) end

  local blocks = {}
  if #title > 0 then
    blocks[#blocks + 1] = pandoc.Para(title)
  end
  if #rest > 0 then
    blocks[#blocks + 1] = pandoc.Para(rest)
  end
  for i = 2, #el.content do blocks[#blocks + 1] = el.content[i] end
  return pandoc.Div(blocks, pandoc.Attr("", { "alert-blue" }))
end

function Link(el)
  local t = el.target
  t = (t:gsub("^_appendix%.md", "appendix.html"))
  t = (t:gsub("^main%.md", "index.html"))
  t = (t:gsub("^%.%./figs/", "figs/"))
  t = (t:gsub("^%.%./src/", "https://github.com/tguens/scratch-rl-notes/blob/main/src/"))
  el.target = t
  return el
end

function Image(el)
  el.src = (el.src:gsub("^%.%./figs/", "figs/"))
  return el
end
