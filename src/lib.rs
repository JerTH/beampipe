
#[macro_use]
pub mod macros;
pub mod ast;
pub mod parse;
pub mod token;
pub mod codegen;


/**
 * Notes
 * 
 * Collider Integration
 * 
 * - Scripts can be run as jobs on the threadpool
 * - Scripts only ever have a read-only copy of the arguments they take
 * - The final resulting value that a script wishes to write back to a component
 *      in collider is emitted as a DatabaseCommand
 * - DatabaseCommands are executed at the discretion of the consumer of collider,
 *      or collider itself. The command queue can contain more than just script
 *      mutations, and it can be executed at the beginning of an update, at the
 *      end of an update, or somewhere in between
 * - Component mutating database commands might be sorted into a separate set of
 *      queues predicated by the column their update should be applied to, or
 *      the component type which they update. In this way, when a selection
 *      tries to generate an iterator for a column, its command buffer can first
 *      be flushed and any pending changes applied.
 * 
 * 
 */
struct __NOTES;