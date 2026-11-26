//! Wrapping text field (prompt, seed). One paragraph: pasted newlines become spaces.
//! Based on gpui's examples/input.rs, extended to soft-wrapped multi-line layout.
use gpui::{
    App, AvailableSpace, Bounds, ClipboardItem, Context, CursorStyle, ElementId, ElementInputHandler, Entity, EntityInputHandler, FocusHandle, Focusable,
    GlobalElementId, Hsla, KeyBinding, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PaintQuad, Pixels, Point, ScrollHandle,
    SharedString, Style, TextAlign, TextRun, UTF16Selection, UnderlineStyle, Window, WrappedLine, actions, div, fill, point, prelude::*, px, relative, rgb,
    rgba, size,
};
use std::ops::Range;
use unicode_segmentation::*;

use crate::theme::*;

actions!(
    text_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        Up,
        Down,
        SelectLeft,
        SelectRight,
        SelectUp,
        SelectDown,
        SelectAll,
        Home,
        End,
        Paste,
        Cut,
        Copy,
        WordLeft,
        WordRight,
        SelectWordLeft,
        SelectWordRight,
        DeleteWordLeft,
        DeleteWordRight,
        SelectHome,
        SelectEnd,
        DocStart,
        DocEnd,
        SelectDocStart,
        SelectDocEnd,
        Undo,
        Redo
    ]
);

pub fn bind_keys(cx: &mut App) {
    let c = Some("TextInput");
    cx.bind_keys([
        KeyBinding::new("backspace", Backspace, c),
        KeyBinding::new("delete", Delete, c),
        KeyBinding::new("left", Left, c),
        KeyBinding::new("right", Right, c),
        KeyBinding::new("up", Up, c),
        KeyBinding::new("down", Down, c),
        KeyBinding::new("shift-left", SelectLeft, c),
        KeyBinding::new("shift-right", SelectRight, c),
        KeyBinding::new("shift-up", SelectUp, c),
        KeyBinding::new("shift-down", SelectDown, c),
        KeyBinding::new("ctrl-a", SelectAll, c),
        KeyBinding::new("ctrl-v", Paste, c),
        KeyBinding::new("ctrl-c", Copy, c),
        KeyBinding::new("ctrl-x", Cut, c),
        KeyBinding::new("home", Home, c),
        KeyBinding::new("end", End, c),
        // Windows-style word and document navigation (alt- also works, like macOS)
        KeyBinding::new("ctrl-left", WordLeft, c),
        KeyBinding::new("ctrl-right", WordRight, c),
        KeyBinding::new("alt-left", WordLeft, c),
        KeyBinding::new("alt-right", WordRight, c),
        KeyBinding::new("ctrl-shift-left", SelectWordLeft, c),
        KeyBinding::new("ctrl-shift-right", SelectWordRight, c),
        KeyBinding::new("alt-shift-left", SelectWordLeft, c),
        KeyBinding::new("alt-shift-right", SelectWordRight, c),
        KeyBinding::new("ctrl-backspace", DeleteWordLeft, c),
        KeyBinding::new("alt-backspace", DeleteWordLeft, c),
        KeyBinding::new("ctrl-delete", DeleteWordRight, c),
        KeyBinding::new("alt-delete", DeleteWordRight, c),
        KeyBinding::new("shift-backspace", Backspace, c),
        KeyBinding::new("shift-delete", Delete, c),
        KeyBinding::new("shift-home", SelectHome, c),
        KeyBinding::new("shift-end", SelectEnd, c),
        KeyBinding::new("ctrl-home", DocStart, c),
        KeyBinding::new("ctrl-end", DocEnd, c),
        KeyBinding::new("ctrl-shift-home", SelectDocStart, c),
        KeyBinding::new("ctrl-shift-end", SelectDocEnd, c),
        KeyBinding::new("ctrl-up", DocStart, c),
        KeyBinding::new("ctrl-down", DocEnd, c),
        KeyBinding::new("ctrl-z", Undo, c),
        KeyBinding::new("ctrl-y", Redo, c),
        KeyBinding::new("ctrl-shift-z", Redo, c),
    ]);
}

pub struct TextInput {
    pub focus_handle: FocusHandle,
    content: SharedString,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_layout: Option<WrappedLine>,
    last_bounds: Option<Bounds<Pixels>>,
    is_selecting: bool,
    min_lines: usize,
    max_lines: usize, // 0 = grows without limit; else scrolls past it
    scroll: ScrollHandle,
    follow_cursor: bool,
    pad_right: f32,
    mono: bool,
    bare: bool,
    undo: Vec<(SharedString, Range<usize>)>,
    redo: Vec<(SharedString, Range<usize>)>,
    typing: bool, // the last edit was a typed character: consecutive ones undo as one step (until a space)
}

const LINE_H: f32 = 20.;

impl TextInput {
    pub fn new(cx: &mut Context<Self>, placeholder: &str, min_lines: usize) -> Self {
        Self {
            focus_handle: cx.focus_handle().tab_stop(true),
            content: "".into(),
            placeholder: placeholder.to_string().into(),
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            last_layout: None,
            last_bounds: None,
            is_selecting: false,
            min_lines,
            max_lines: 0,
            scroll: ScrollHandle::new(),
            follow_cursor: false,
            pad_right: 11.,
            mono: false,
            bare: false,
            undo: vec![],
            redo: vec![],
            typing: false,
        }
    }

    /// Visible height between min and max lines; past max the field scrolls and keeps the cursor in view.
    pub fn set_lines(&mut self, min: usize, max: usize, cx: &mut Context<Self>) {
        self.min_lines = min;
        self.max_lines = max;
        self.follow_cursor = true;
        cx.notify();
    }

    /// Room on the right for an overlay button (the expand icon).
    pub fn pad_right(mut self, px_: f32) -> Self {
        self.pad_right = px_;
        self
    }

    pub fn with_max_lines(mut self, max: usize) -> Self {
        self.max_lines = max;
        self
    }

    pub fn mono(mut self) -> Self {
        self.mono = true;
        self
    }

    /// No box of its own: for a field that sits inside a search bar.
    pub fn bare(mut self) -> Self {
        self.bare = true;
        self
    }

    pub fn text(&self) -> String {
        self.content.to_string()
    }

    pub fn set_text(&mut self, text: &str, cx: &mut Context<Self>) {
        self.snapshot(false);
        let t = clean(text);
        self.selected_range = t.len()..t.len();
        self.content = t.into();
        self.marked_range = None;
        cx.notify();
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.start, cx)
        }
    }
    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.selected_range.end), cx);
        } else {
            self.move_to(self.selected_range.end, cx)
        }
    }
    fn up(&mut self, _: &Up, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.vertical(-1.);
        self.move_to(i, cx)
    }
    fn down(&mut self, _: &Down, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.vertical(1.);
        self.move_to(i, cx)
    }
    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }
    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }
    fn select_up(&mut self, _: &SelectUp, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.vertical(-1.);
        self.select_to(i, cx)
    }
    fn select_down(&mut self, _: &SelectDown, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.vertical(1.);
        self.select_to(i, cx)
    }
    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
        self.select_to(self.content.len(), cx)
    }
    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.line_edge(false);
        self.move_to(i, cx);
    }
    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.line_edge(true);
        self.move_to(i, cx);
    }
    fn word_left(&mut self, _: &WordLeft, _: &mut Window, cx: &mut Context<Self>) {
        let i = if self.selected_range.is_empty() { prev_word(&self.content, self.cursor_offset()) } else { self.selected_range.start };
        self.move_to(i, cx)
    }
    fn word_right(&mut self, _: &WordRight, _: &mut Window, cx: &mut Context<Self>) {
        let i = if self.selected_range.is_empty() { next_word(&self.content, self.cursor_offset()) } else { self.selected_range.end };
        self.move_to(i, cx)
    }
    fn select_word_left(&mut self, _: &SelectWordLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(prev_word(&self.content, self.cursor_offset()), cx)
    }
    fn select_word_right(&mut self, _: &SelectWordRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(next_word(&self.content, self.cursor_offset()), cx)
    }
    fn delete_word_left(&mut self, _: &DeleteWordLeft, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(prev_word(&self.content, self.cursor_offset()), cx)
        }
        self.typing = false;
        self.replace_text_in_range(None, "", window, cx)
    }
    fn delete_word_right(&mut self, _: &DeleteWordRight, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(next_word(&self.content, self.cursor_offset()), cx)
        }
        self.typing = false;
        self.replace_text_in_range(None, "", window, cx)
    }
    fn select_home(&mut self, _: &SelectHome, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.line_edge(false);
        self.select_to(i, cx)
    }
    fn select_end(&mut self, _: &SelectEnd, _: &mut Window, cx: &mut Context<Self>) {
        let i = self.line_edge(true);
        self.select_to(i, cx)
    }
    fn doc_start(&mut self, _: &DocStart, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx)
    }
    fn doc_end(&mut self, _: &DocEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.content.len(), cx)
    }
    fn select_doc_start(&mut self, _: &SelectDocStart, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(0, cx)
    }
    fn select_doc_end(&mut self, _: &SelectDocEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.content.len(), cx)
    }
    fn undo_action(&mut self, _: &Undo, _: &mut Window, cx: &mut Context<Self>) {
        if let Some((text, sel)) = self.undo.pop() {
            self.redo.push((self.content.clone(), self.selected_range.clone()));
            self.restore(text, sel, cx);
        }
    }
    fn redo_action(&mut self, _: &Redo, _: &mut Window, cx: &mut Context<Self>) {
        if let Some((text, sel)) = self.redo.pop() {
            self.undo.push((self.content.clone(), self.selected_range.clone()));
            self.restore(text, sel, cx);
        }
    }
    fn restore(&mut self, text: SharedString, sel: Range<usize>, cx: &mut Context<Self>) {
        self.content = text;
        self.selected_range = sel.start.min(self.content.len())..sel.end.min(self.content.len());
        self.selection_reversed = false;
        self.marked_range = None;
        self.typing = false;
        self.follow_cursor = true;
        cx.notify();
    }
    /// Saves the state before an edit; `typed` = one typed character, merged with the ones typed just before it.
    fn snapshot(&mut self, typed: bool) {
        if !(typed && self.typing) {
            self.undo.push((self.content.clone(), self.selected_range.clone()));
            if self.undo.len() > 200 {
                self.undo.remove(0);
            }
        }
        self.typing = typed;
        self.redo.clear();
    }
    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        self.typing = false;
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }
    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        self.typing = false;
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }
    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        match cx.read_from_clipboard().and_then(|item| item.text()) {
            Some(text) => {
                self.typing = false;
                self.replace_text_in_range(None, &clean(&text), window, cx)
            }
            None => cx.propagate(), // an image: the studio takes it as the source
        }
    }
    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(self.content[self.selected_range.clone()].to_string()));
        }
    }
    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(self.content[self.selected_range.clone()].to_string()));
            self.typing = false;
            self.replace_text_in_range(None, "", window, cx)
        }
    }

    fn on_mouse_down(&mut self, event: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus_handle);
        cx.stop_propagation();
        self.is_selecting = true;
        let i = self.index_for_mouse_position(event.position);
        if event.modifiers.shift {
            self.select_to(i, cx);
        } else if event.click_count == 2 {
            // double click: the word under the cursor
            let start = self.content[..i].unicode_word_indices().next_back().map(|(s, w)| if s + w.len() >= i { s } else { i }).unwrap_or(i);
            let end = self.content[i..].unicode_word_indices().next().map(|(s, w)| if s == 0 { i + w.len() } else { i }).unwrap_or(i);
            self.selected_range = start..end;
            self.selection_reversed = false;
            cx.notify();
        } else {
            self.move_to(i, cx)
        }
    }
    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }
    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            self.select_to(self.index_for_mouse_position(event.position), cx);
        }
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.follow_cursor = true;
        self.selected_range = offset..offset;
        cx.notify()
    }
    fn cursor_offset(&self) -> usize {
        if self.selection_reversed { self.selected_range.start } else { self.selected_range.end }
    }
    fn index_at(&self, local: Point<Pixels>) -> usize {
        let Some(line) = self.last_layout.as_ref() else { return 0 };
        if self.content.is_empty() {
            return 0;
        }
        if local.y < px(0.) {
            return 0;
        }
        if local.y >= px(LINE_H * (line.wrap_boundaries().len() + 1) as f32) {
            return self.content.len();
        }
        match line.closest_index_for_position(local, px(LINE_H)) {
            Ok(i) | Err(i) => i.min(self.content.len()),
        }
    }
    fn index_for_mouse_position(&self, position: Point<Pixels>) -> usize {
        let Some(bounds) = self.last_bounds else { return 0 };
        if position.y > bounds.bottom() {
            return self.content.len();
        }
        self.index_at(point(position.x - bounds.left(), position.y - bounds.top()))
    }
    fn vertical(&self, dir: f32) -> usize {
        let cur = self.cursor_offset();
        let Some(line) = self.last_layout.as_ref() else { return cur };
        let Some(p) = line.position_for_index(cur, px(LINE_H)) else { return cur };
        let y = p.y + px(LINE_H * dir) + px(LINE_H / 2.);
        if y < px(0.) {
            return 0;
        }
        let lines = line.wrap_boundaries().len() + 1;
        if y > px(LINE_H * lines as f32) {
            return self.content.len();
        }
        self.index_at(point(p.x, y))
    }
    fn line_edge(&self, end: bool) -> usize {
        let cur = self.cursor_offset();
        let Some(line) = self.last_layout.as_ref() else { return if end { self.content.len() } else { 0 } };
        let Some(p) = line.position_for_index(cur, px(LINE_H)) else { return cur };
        self.index_at(point(if end { px(100000.) } else { px(0.) }, p.y + px(LINE_H / 2.)))
    }
    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.follow_cursor = true;
        if self.selection_reversed {
            self.selected_range.start = offset
        } else {
            self.selected_range.end = offset
        };
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify()
    }
    fn offset_from_utf16(&self, offset: usize) -> usize {
        let (mut utf8, mut utf16) = (0, 0);
        for ch in self.content.chars() {
            if utf16 >= offset {
                break;
            }
            utf16 += ch.len_utf16();
            utf8 += ch.len_utf8();
        }
        utf8
    }
    fn offset_to_utf16(&self, offset: usize) -> usize {
        let (mut utf16, mut utf8) = (0, 0);
        for ch in self.content.chars() {
            if utf8 >= offset {
                break;
            }
            utf8 += ch.len_utf8();
            utf16 += ch.len_utf16();
        }
        utf16
    }
    fn range_to_utf16(&self, r: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(r.start)..self.offset_to_utf16(r.end)
    }
    fn range_from_utf16(&self, r: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(r.start)..self.offset_from_utf16(r.end)
    }
    fn previous_boundary(&self, offset: usize) -> usize {
        self.content.grapheme_indices(true).rev().find_map(|(i, _)| (i < offset).then_some(i)).unwrap_or(0)
    }
    fn next_boundary(&self, offset: usize) -> usize {
        self.content.grapheme_indices(true).find_map(|(i, _)| (i > offset).then_some(i)).unwrap_or(self.content.len())
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Class {
    Space,
    Word,
    Punct,
}

fn class(c: char) -> Class {
    if c.is_whitespace() {
        Class::Space
    } else if c.is_alphanumeric() || c == '_' {
        Class::Word
    } else {
        Class::Punct
    }
}

/// Ctrl+Left: start of the word before `offset` (spaces skipped first; a run of punctuation counts as a word).
fn prev_word(s: &str, offset: usize) -> usize {
    let mut chars = s[..offset.min(s.len())].char_indices().rev().peekable();
    while chars.peek().is_some_and(|(_, c)| class(*c) == Class::Space) {
        chars.next();
    }
    let Some(&(first, c0)) = chars.peek() else { return 0 };
    let k = class(c0);
    let mut start = first;
    while let Some(&(i, c)) = chars.peek() {
        if class(c) != k {
            break;
        }
        start = i;
        chars.next();
    }
    start
}

/// Ctrl+Right (Windows): past the current word and the spaces after it = start of the next word.
fn next_word(s: &str, offset: usize) -> usize {
    let offset = offset.min(s.len());
    let mut chars = s[offset..].char_indices().map(|(i, c)| (i + offset, c)).peekable();
    if let Some(&(_, c0)) = chars.peek() {
        let k = class(c0);
        if k != Class::Space {
            while chars.peek().is_some_and(|(_, c)| class(*c) == k) {
                chars.next();
            }
        }
    }
    while chars.peek().is_some_and(|(_, c)| class(*c) == Class::Space) {
        chars.next();
    }
    chars.peek().map(|(i, _)| *i).unwrap_or(s.len())
}

fn clean(t: &str) -> String {
    t.replace("\r\n", " ").replace(['\n', '\r', '\t'], " ")
}

impl EntityInputHandler for TextInput {
    fn text_for_range(&mut self, range_utf16: Range<usize>, actual: &mut Option<Range<usize>>, _: &mut Window, _: &mut Context<Self>) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual.replace(self.range_to_utf16(&range));
        Some(self.content[range].to_string())
    }
    fn selected_text_range(&mut self, _: bool, _: &mut Window, _: &mut Context<Self>) -> Option<UTF16Selection> {
        Some(UTF16Selection { range: self.range_to_utf16(&self.selected_range), reversed: self.selection_reversed })
    }
    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range.as_ref().map(|r| self.range_to_utf16(r))
    }
    fn unmark_text(&mut self, _: &mut Window, _: &mut Context<Self>) {
        self.marked_range = None;
    }
    fn replace_text_in_range(&mut self, range_utf16: Option<Range<usize>>, new_text: &str, _: &mut Window, cx: &mut Context<Self>) {
        let new_text = clean(new_text);
        let range = range_utf16.as_ref().map(|r| self.range_from_utf16(r)).or(self.marked_range.clone()).unwrap_or(self.selected_range.clone());
        // one undo step per typed word: single characters merge until a space or any other kind of edit
        let typed = range.is_empty() && new_text.chars().count() == 1 && !new_text.starts_with(' ');
        self.snapshot(typed);
        self.content = (self.content[0..range.start].to_owned() + &new_text + &self.content[range.end..]).into();
        self.selected_range = range.start + new_text.len()..range.start + new_text.len();
        self.follow_cursor = true;
        self.marked_range.take();
        cx.notify();
    }
    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected: Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16.as_ref().map(|r| self.range_from_utf16(r)).or(self.marked_range.clone()).unwrap_or(self.selected_range.clone());
        self.content = (self.content[0..range.start].to_owned() + new_text + &self.content[range.end..]).into();
        self.marked_range = if new_text.is_empty() { None } else { Some(range.start..range.start + new_text.len()) };
        self.selected_range = new_selected
            .as_ref()
            .map(|r| self.range_from_utf16(r))
            .map(|r| r.start + range.start..r.end + range.start)
            .unwrap_or_else(|| range.start + new_text.len()..range.start + new_text.len());
        cx.notify();
    }
    fn bounds_for_range(&mut self, range_utf16: Range<usize>, bounds: Bounds<Pixels>, _: &mut Window, _: &mut Context<Self>) -> Option<Bounds<Pixels>> {
        let line = self.last_layout.as_ref()?;
        let range = self.range_from_utf16(&range_utf16);
        let p = line.position_for_index(range.start, px(LINE_H))?;
        Some(Bounds::new(point(bounds.left() + p.x, bounds.top() + p.y), size(px(2.), px(LINE_H))))
    }
    fn character_index_for_point(&mut self, p: Point<Pixels>, _: &mut Window, _: &mut Context<Self>) -> Option<usize> {
        let b = self.last_bounds?;
        Some(self.offset_to_utf16(self.index_at(point(p.x - b.left(), p.y - b.top()))))
    }
}

struct TextElement {
    input: Entity<TextInput>,
}

struct Prepaint {
    line: Option<WrappedLine>,
    cursor: Option<PaintQuad>,
    selection: Vec<PaintQuad>,
}

impl IntoElement for TextElement {
    type Element = Self;
    fn into_element(self) -> Self::Element {
        self
    }
}

fn shape(input: &TextInput, window: &mut Window, width: Option<Pixels>) -> Option<WrappedLine> {
    let style = window.text_style();
    let (text, color): (SharedString, Hsla) =
        if input.content.is_empty() { (input.placeholder.clone(), rgb(MUTED_2()).into()) } else { (input.content.clone(), style.color) };
    if text.is_empty() {
        return None;
    }
    let run = TextRun { len: text.len(), font: style.font(), color, background_color: None, underline: None, strikethrough: None };
    let runs = match input.marked_range.as_ref().filter(|_| !input.content.is_empty()) {
        Some(m) => vec![
            TextRun { len: m.start, ..run.clone() },
            TextRun { len: m.end - m.start, underline: Some(UnderlineStyle { color: Some(run.color), thickness: px(1.), wavy: false }), ..run.clone() },
            TextRun { len: text.len() - m.end, ..run },
        ]
        .into_iter()
        .filter(|r| r.len > 0)
        .collect(),
        None => vec![run],
    };
    let font_size = style.font_size.to_pixels(window.rem_size());
    window.text_system().shape_text(text, font_size, &runs, width, None).ok()?.into_iter().next()
}

impl Element for TextElement {
    type RequestLayoutState = ();
    type PrepaintState = Prepaint;
    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }
    fn request_layout(&mut self, _: Option<&GlobalElementId>, _: Option<&gpui::InspectorElementId>, window: &mut Window, _cx: &mut App) -> (LayoutId, ()) {
        // height = wrapped lines at the width the layout actually gives us (measured, not last frame's width)
        let input = self.input.clone();
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        let id = window.request_measured_layout(style, move |known, available, window, cx| {
            let width = known.width.or(match available.width {
                AvailableSpace::Definite(w) => Some(w),
                _ => None,
            });
            let input = input.read(cx);
            let lines = width.and_then(|w| shape(input, window, Some(w))).map(|l| l.wrap_boundaries().len() + 1).unwrap_or(1);
            size(width.unwrap_or(px(0.)), px(LINE_H * lines.max(input.min_lines) as f32))
        });
        (id, ())
    }
    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> Prepaint {
        let input = self.input.read(cx);
        let line = shape(input, window, Some(bounds.size.width));
        let lh = px(LINE_H);
        let at = |i: usize| -> Point<Pixels> {
            if input.content.is_empty() {
                return point(px(0.), px(0.));
            }
            line.as_ref().and_then(|l| l.position_for_index(i, lh)).unwrap_or(point(px(0.), px(0.)))
        };
        let r = input.selected_range.clone();
        let mut selection = vec![];
        let mut cursor = None;
        if r.is_empty() {
            let p = at(input.cursor_offset());
            cursor = Some(fill(Bounds::new(point(bounds.left() + p.x, bounds.top() + p.y + px(2.)), size(px(1.5), lh - px(4.))), rgb(ACCENT())));
        } else {
            let (a, b) = (at(r.start), at(r.end));
            let sel = rgba(0xe5a95f50);
            if a.y == b.y {
                selection
                    .push(fill(Bounds::from_corners(point(bounds.left() + a.x, bounds.top() + a.y), point(bounds.left() + b.x, bounds.top() + b.y + lh)), sel));
            } else {
                selection.push(fill(Bounds::from_corners(point(bounds.left() + a.x, bounds.top() + a.y), point(bounds.right(), bounds.top() + a.y + lh)), sel));
                if b.y > a.y + lh {
                    selection.push(fill(Bounds::from_corners(point(bounds.left(), bounds.top() + a.y + lh), point(bounds.right(), bounds.top() + b.y)), sel));
                }
                selection.push(fill(Bounds::from_corners(point(bounds.left(), bounds.top() + b.y), point(bounds.left() + b.x, bounds.top() + b.y + lh)), sel));
            }
        }
        Prepaint { line, cursor, selection }
    }
    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        prepaint: &mut Prepaint,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus = self.input.read(cx).focus_handle.clone();
        window.handle_input(&focus, ElementInputHandler::new(bounds, self.input.clone()), cx);
        for q in prepaint.selection.drain(..) {
            window.paint_quad(q);
        }
        if let Some(line) = prepaint.line.as_ref() {
            let _ = line.paint(bounds.origin, px(LINE_H), TextAlign::Left, None, window, cx);
        }
        if focus.is_focused(window)
            && let Some(c) = prepaint.cursor.take()
        {
            window.paint_quad(c);
        }
        let line = prepaint.line.take();
        self.input.update(cx, |input, cx| {
            // keep the cursor line inside the scrolled viewport after typing / keyboard moves
            if input.follow_cursor && input.max_lines > 0 {
                input.follow_cursor = false;
                let cur = input.cursor_offset();
                let y = if input.content.is_empty() { Some(px(0.)) } else { line.as_ref().and_then(|l| l.position_for_index(cur, px(LINE_H))).map(|p| p.y) };
                let view = input.scroll.bounds();
                if let Some(y) = y {
                    let top = bounds.top() + y;
                    let mut off = input.scroll.offset();
                    if top < view.top() {
                        off.y += view.top() - top;
                    } else if top + px(LINE_H) > view.bottom() {
                        off.y -= top + px(LINE_H) - view.bottom();
                    }
                    if off != input.scroll.offset() {
                        input.scroll.set_offset(off);
                        cx.notify();
                    }
                }
            }
            input.last_layout = if input.content.is_empty() { None } else { line };
            input.last_bounds = Some(bounds);
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let focused = self.focus_handle.is_focused(window);
        div()
            .key_context("TextInput")
            .track_focus(&self.focus_handle)
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::up))
            .on_action(cx.listener(Self::down))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_up))
            .on_action(cx.listener(Self::select_down))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::word_left))
            .on_action(cx.listener(Self::word_right))
            .on_action(cx.listener(Self::select_word_left))
            .on_action(cx.listener(Self::select_word_right))
            .on_action(cx.listener(Self::delete_word_left))
            .on_action(cx.listener(Self::delete_word_right))
            .on_action(cx.listener(Self::select_home))
            .on_action(cx.listener(Self::select_end))
            .on_action(cx.listener(Self::doc_start))
            .on_action(cx.listener(Self::doc_end))
            .on_action(cx.listener(Self::select_doc_start))
            .on_action(cx.listener(Self::select_doc_end))
            .on_action(cx.listener(Self::undo_action))
            .on_action(cx.listener(Self::redo_action))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .w_full()
            .when(!self.bare, |d| {
                d.pl(px(11.)).pr(px(self.pad_right)).py(px(8.)).rounded(px(7.)).bg(rgb(INPUT_BG())).border_1().border_color(rgb(if focused {
                    ACCENT_DIM()
                } else {
                    BORDER()
                }))
            })
            .when(self.bare, |d| d.py(px(6.)).text_size(px(14.)))
            .when(!self.bare, |d| d.text_size(px(13.)))
            .line_height(px(LINE_H))
            .text_color(rgb(TEXT()))
            .when(self.mono, |d| d.font_family(MONO))
            .child(
                div()
                    .id(("text-scroll", cx.entity_id().as_u64() as usize))
                    .w_full()
                    .when(self.max_lines > 0, |d| d.max_h(px(LINE_H * self.max_lines as f32)).overflow_y_scroll().track_scroll(&self.scroll))
                    .child(TextElement { input: cx.entity() }),
            )
    }
}

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{next_word, prev_word};

    #[test]
    fn word_jumps_like_windows() {
        let s = "a short silver wand, gold";
        assert_eq!(prev_word(s, s.len()), 21); // gold
        assert_eq!(prev_word(s, 21), 19); // the comma
        assert_eq!(prev_word(s, 19), 15); // wand
        assert_eq!(prev_word(s, 2), 0);
        assert_eq!(prev_word(s, 0), 0);
        assert_eq!(next_word(s, 0), 2); // past "a " to short
        assert_eq!(next_word(s, 2), 8);
        assert_eq!(next_word(s, 15), 19); // wand to the comma
        assert_eq!(next_word(s, 21), s.len());
        let u = "espada \u{f1}and\u{fa} r\u{e1}pida";
        let second = "espada ".len();
        let third = "espada \u{f1}and\u{fa} ".len();
        assert_eq!(next_word(u, 0), second);
        assert_eq!(next_word(u, second), third);
        assert_eq!(prev_word(u, u.len()), third);
    }
}
