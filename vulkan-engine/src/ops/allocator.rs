/// Generic slice allocator in any fixed-capacity buffer.
/// 
/// ## Fields
/// 
/// - `free_slots` (`Vec<u32>`) - available slot indexes.
/// - `slot_size` (`u32`) - Size of a slot.
/// - `capacity` (`u32`) - total number of slots.
pub struct SlotAllocator {
    free_slots: Vec<u32>,
    slot_size: u32,
}

impl SlotAllocator {
    pub fn new(capacity: u32, slot_size: u32) -> Self {
        Self {
            free_slots: (0..capacity).rev().collect(), // reverse to have a stack instead of a pile with pop()
            slot_size,
        }
    }

    /// Return the first available slice.
    pub fn allocate(&mut self) -> Option<u32> {
        self.free_slots.pop().map(|slot| slot * self.slot_size)
    }

    /// From an **offset** retrieve the slot index and place it in the free_slots pile.
    pub fn free(&mut self, offset: u32) {
        self.free_slots.push(offset / self.slot_size)
    }
}