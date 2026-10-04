"""Curriculum prompt and <PAGE>/<GOAL> proposal parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_TARGET_ACTIONS: int = 6

_WEB_CURRICULUM_SYSTEM_TEMPLATE = """\
You are a CURRICULUM AGENT for a learned web WORLD MODEL. You invent a self-contained WEB task for a \
browsing AGENT to attempt: you design the initial page (as an accessibility tree, in the world model's \
style) and a goal, and the world model then rolls out the agent's actions from that page.

The agent interacts with the page using this browser action set (ONE action per turn):
- click(bid, button, modifiers)   - click a DOM element by its id
- fill(bid, text, press_enter)    - type text into an input field
- select_option(bid, options)     - select from a dropdown / combobox
- hover(bid)                      - hover over an element
- mouse_move(x, y) / mouse_click(x, y, button) / mouse_down(x, y) / mouse_up(x, y)
- keyboard_press(key) / keyboard_type(text)
- scroll(dx, dy) / goto(url) / go_back() / go_forward()
- tab_new() / tab_close() / tab_focus(index)
- send_msg_to_user(text) / noop(wait_ms) / infeasible(reason)

Output EXACTLY this format, with nothing else:

<PAGE>
RootWebArea '<page title>', focused
\t[<bid>] <role> '<label>', <properties>
\t\t[<bid>] <role> '<label>', <properties>
\t...
</PAGE>
<GOAL>
<one short natural-language task sentence>
</GOAL>

Rules:
- The PAGE is a realistic WEB page in the world model's accessibility-tree style — ANY kind of site
  (news, blog, docs, settings, dashboard, forum, search results, a shop, …), NOT limited to one domain.
  Start with a RootWebArea title line, then nested `[bid] role 'label', <properties>` lines; use tab
  indentation to reflect structure (navigation, sections, lists). Include realistic page chrome (nav
  links, headings, buttons) plus the elements the task needs.
- bids are small unique integers ([1], [12], [42]). Roles include: textbox, button, link, combobox,
  listitem, heading, paragraph, navigation, section, image. Properties may include: clickable, visible,
  focused, editable, value='...'.
- Reference ONLY elements you wrote in <PAGE>. DO NOT mention bid numbers in the goal — refer to
  elements by their visible label.
- Give the PAGE roughly 15-30 elements: enough material for a multi-step task (fields to fill, rows to
  compare, plausible distractor buttons/fields the goal tells the agent to leave alone), still a tree a
  reader can hold in their head.
- DIFFICULTY TARGET: the goal must take about N_ACTIONS browser actions to complete — count one action
  per click / fill / select_option / hover / scroll / keyboard_press. Design the PAGE so that many
  actions are genuinely REQUIRED (that many fields to fill, that many controls to touch), not padded.
  A task a competent agent finishes in 1-3 actions is TOO EASY and is worthless here. Reliable ways to
  reach N_ACTIONS actions: several form fields whose correct values are written elsewhere on the same
  page (a summary/profile panel) and must be copied across; a value that must be derived by comparing
  the rows of a catalog/table before it is entered; an arithmetic value (unit price times quantity); a
  prescribed ORDER of steps mixing fills, a dropdown selection, a hover or scroll, and a final submit.
- The goal must be solvable with the action set above using only elements on the page, and be
  UNAMBIGUOUS — a reader should know exactly what "done" means. It must have EXACTLY ONE correct
  answer, but finding that answer MAY require reading and comparing values present on the page
  (e.g. "the cheapest in-stock model rated above 4.5"). Whenever the goal turns on a comparison, make
  the page pick a unique winner: no ties on the deciding attribute, and every filter the goal names
  ("in stock", "at least 512GB") must really rule the other candidates out.
- Name the exact field or control for every step, and say what to leave alone (name the distractor
  fields or buttons the agent must NOT touch), so that one exact sequence of actions counts as correct.

Here are example proposals:

<PAGE>
RootWebArea 'TicketHub — Ticket Verification', focused
\t[1] navigation ''
\t\t[2] link 'Pricing'
\t\t[3] link 'Support'
\t[4] heading 'Ticket Verification'
\t[5] section 'Order Summary'
\t\t[6] paragraph 'Attendee name: Theo Novak'
\t\t[7] paragraph 'Order reference: TH-29109'
\t\t[8] paragraph 'Total paid: $154.25'
\t\t[9] paragraph 'Venue city: Redcliff'
\t\t[10] paragraph 'Seat block: C'
\t[11] heading 'Enter the details'
\t[12] textbox 'Attendee name', editable, value=''
\t[13] textbox 'Order reference', editable, value=''
\t[14] textbox 'Total paid', editable, value=''
\t[15] textbox 'Venue city', editable, value=''
\t[16] textbox 'Promo code', editable, value=''
\t[17] combobox 'Delivery method'
\t\t[18] listitem 'Email'
\t\t[19] listitem 'Print at home'
\t\t[20] listitem 'Box office pickup'
\t[21] button 'Confirm tickets'
\t[22] button 'Clear form'
\t[23] button 'Cancel'
</PAGE>
<GOAL>
Copy the attendee name, order reference, total paid and venue city shown under Order Summary into the matching boxes below it, set the delivery method to box office pickup, then click Confirm tickets; leave the promo code box empty.
</GOAL>

<PAGE>
RootWebArea 'FieldGear — Order Configuration', focused
\t[1] heading 'Camp Stoves'
\t[2] section 'Catalog'
\t\t[3] listitem 'SKU TR-118 — 2.4 kg — $189 — rating 4.6 — In Stock'
\t\t[4] listitem 'SKU QN-540 — 1.9 kg — $164 — rating 4.8 — Out of Stock'
\t\t[5] listitem 'SKU BW-273 — 1.7 kg — $205 — rating 4.7 — In Stock'
\t\t[6] listitem 'SKU LM-902 — 2.1 kg — $150 — rating 4.1 — In Stock'
\t[7] heading 'Order Form'
\t[8] textbox 'Model SKU', editable, value=''
\t[9] textbox 'Unit price', editable, value=''
\t[10] textbox 'Quantity', editable, value=''
\t[11] textbox 'Order total', editable, value=''
\t[12] textbox 'Gift note', editable, value=''
\t[13] combobox 'Shipping speed'
\t\t[14] listitem 'Ground'
\t\t[15] listitem 'Two-day'
\t\t[16] listitem 'Overnight'
\t[17] button 'Place order'
\t[18] button 'Save draft'
\t[19] button 'Cancel'
</PAGE>
<GOAL>
Order 3 units of the lightest in-stock stove rated above 4.5: put its SKU in the model SKU box, its price without the dollar sign in the unit price box, 3 in the quantity box, and the unit price times the quantity in the order total box, set the shipping speed to two-day, then click Place order; leave the gift note empty.
</GOAL>

<PAGE>
RootWebArea 'Meridian — Profile & Preferences', focused
\t[1] navigation ''
\t\t[2] link 'Overview'
\t\t[3] link 'Billing'
\t[4] heading 'Profile & Preferences'
\t[5] section 'Profile on record'
\t\t[6] paragraph 'Recorded phone: 555-505-2696'
\t\t[7] paragraph 'Recorded job title: Director'
\t\t[8] paragraph 'Recorded region: Oceania'
\t\t[9] paragraph 'Preferred theme: Dark'
\t[10] heading 'Account details'
\t[11] textbox 'Account phone', editable, value=''
\t[12] textbox 'Account job title', editable, value=''
\t[13] textbox 'Account region', editable, value=''
\t[14] textbox 'Account company', editable, value=''
\t[15] combobox 'Theme'
\t\t[16] listitem 'Light'
\t\t[17] listitem 'Dark'
\t\t[18] listitem 'System'
\t[19] button 'Help tips'
\t[20] button 'Save profile'
\t[21] button 'Discard changes'
</PAGE>
<GOAL>
Do these steps in this order: hover over the help tips button, copy the recorded phone into the account phone box, copy the recorded job title into the account job title box, scroll down the page, copy the recorded region into the account region box, set the theme dropdown to the preferred theme shown on record, press the Tab key, then click Save profile; leave the account company box empty.
</GOAL>
"""


WEB_CURRICULUM_SYSTEM: str = _WEB_CURRICULUM_SYSTEM_TEMPLATE.replace("N_ACTIONS", str(DEFAULT_TARGET_ACTIONS))

# Proposal i is conditioned on domain WEB_DOMAINS[i % len(WEB_DOMAINS)].
WEB_DOMAINS = [
    "a NEWS or blog homepage — open one specific named article, then fill and submit the comment or "
    "subscribe form on it, copying the reader details already shown on the page",
    "a SETTINGS or preferences page — set several named options (at least one combobox) to the values "
    "listed in a 'current profile' panel, then click Save",
    "a SEARCH page — fill the query box, submit, then take the one result matching a stated constraint "
    "and copy its details into the form beside the results",
    "a DOCS or help site — use the navigation menu to reach one named page, then complete the multi-"
    "field feedback form there using values shown in that page's info box",
    "a longer FORM (checkout, registration, shipping) — copy 4–6 named fields from a summary panel on "
    "the same page, set a dropdown, then submit; some named fields must be left blank",
    "an app DASHBOARD — open the named section, read the metrics shown there, and enter those values "
    "plus a derived total into the report form before submitting",
    "a SHOP catalog — compare the listed models on price/stock/rating, then order a stated quantity of "
    "the one meeting the constraint: SKU, unit price, quantity, order total, shipping option, submit",
    "a FORUM or social feed — open one specific named thread, then complete a report/reply form whose "
    "fields must be copied from the thread's metadata",
    "a DERIVE page — a billing/profile panel plus a set of empty fields with matching names: copy each "
    "named value across in the exact order the goal lists, leaving the other named fields blank",
    "a CONSTRAINT page — a table of rows with several attributes; the goal names filters that leave "
    "exactly one row, whose values (plus an arithmetic total) go into the order form",
    "a LONGMIX page — a prescribed ordered sequence mixing fills copied off the page, a dropdown, a "
    "hover or scroll, a keypress, and a final submit",
    "an INVENTORY or booking page — pick the single slot/item satisfying the stated constraints, fill "
    "the reservation form from the details shown for it, then confirm",
]


_PAGE_RE   = re.compile(r"<PAGE>\s*(.*?)\s*</PAGE>", re.DOTALL | re.IGNORECASE)
_GOAL_RE   = re.compile(r"<GOAL>\s*(.*?)\s*</GOAL>", re.DOTALL | re.IGNORECASE)
# Only accessibility-tree element definitions introduce IDs. Numeric references
# inside a label or value are content, so they must not count as duplicate nodes.
_ELEMENT_BID_RE = re.compile(r"(?m)^[ \t]*\[(\d+)\](?=[ \t]+\S)")


@dataclass
class ParsedWebTask:
    page:     str   # the A11y tree the executor will start from
    goal:     str   # the natural-language goal for the executor
    raw:      str   # the full completion (for debugging)
    is_valid: bool  # nonempty PAGE + GOAL, with unique line-level element IDs


def parse_web_completion(text: str) -> ParsedWebTask:
    """Extract <PAGE> + <GOAL> from the curriculum LM's output.

    The format gate requires nonempty PAGE and GOAL and unique numeric IDs on
    accessibility-tree element definitions. Duplicate IDs make browser actions
    ambiguous. Inline numeric references are allowed; semantic solvability is not
    checked here. Invalid proposals receive is_valid=False without repairing IDs."""
    raw = text.strip()
    page_match = _PAGE_RE.search(raw)
    goal_match = _GOAL_RE.search(raw)
    if not (page_match and goal_match):
        return ParsedWebTask(page="", goal="", raw=raw, is_valid=False)
    page = page_match.group(1).strip()
    goal = goal_match.group(1).strip()
    if not (page and goal):
        return ParsedWebTask(page="", goal="", raw=raw, is_valid=False)
    bids = [int(bid) for bid in _ELEMENT_BID_RE.findall(page)]
    return ParsedWebTask(page=page, goal=goal, raw=raw,
                         is_valid=len(bids) == len(set(bids)))

